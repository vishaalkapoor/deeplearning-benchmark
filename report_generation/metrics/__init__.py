#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities to fetch benchmark metrics from Benchmark AI.

The boto3 clients require the environment variables to be set accordingly:
* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_DEFAULT_REGION

This code was modified from code originally written by Pedro Larroy."""

import boto3
import json
import logging
import pickle

from collections import defaultdict
from datetime import datetime,timedelta


events = boto3.client('events')
cw = boto3.client('cloudwatch')

def get_metrics():
    paginator = cw.get_paginator('list_metrics')
    metrics_paginator = paginator.paginate(Namespace='benchmarkai-metrics-prod')
    metrics = metrics_paginator.build_full_result()
    return list(map(lambda x: x['MetricName'], metrics['Metrics']))


def get_rules():
    rules = events.list_rules()['Rules'][2:]
    return list(filter(lambda x: x != 'TriggerNightlyTestStatusCheck', map(lambda x: x['Name'], rules)))


def rules_to_tasks(rules):
    """Get input from the cloudwatch rule input json data
    :returns: a map of rule name to rule input structure
    """
    res = {}
    for rule in rules:
        target = events.list_targets_by_rule(Rule=rule)['Targets'][0]
        if 'Input' in target:
            j = json.loads(target['Input'])
            #print('{} {} {} {}'.format(rule, j.get('task_name'), j.get('num_gpu'), j.get('framework_name')))
            if 'task_name' in j:
                #res.append(j['task_name'])
                res[rule] = j
    return res


def extract_metric(fullmetric, prefix, suffix):
    """Get the metric name from the full metric name string removing prefix and suffix"""
    tail = fullmetric[len(prefix)+1:]
    end = tail.rfind(suffix)-1
    return tail[:end]


def gather_benchmarks(use_cache=False):
    """
    :returns: a dictionary of {'metric': {<metric_name>: value}, ..., 'suffix': } keyed by rule
    """
    benchmarks = {}
    if use_cache:
        logging.info("Loading metrics from cache")
        rules = pickle.load(open("rules.pkl", "rb"))
        metrics = pickle.load(open("metrics.pkl", "rb"))
        rules2tasks = pickle.load(open("rules2tasks.pkl", "rb"))
    else:
        rules = get_rules()
        metrics = get_metrics()
        rules2tasks = rules_to_tasks(rules)
        pickle.dump(rules,open("rules.pkl","wb"))
        pickle.dump(metrics,open("metrics.pkl","wb"))
        pickle.dump(rules2tasks,open("rules2tasks.pkl","wb"))

    logging.info("Got {} rules".format(len(rules)))
    logging.info("Got {} metrics".format(len(metrics)))

    for (rule, task) in rules2tasks.items():
        metric_prefix = '.'.join((task['framework_name'], task['task_name'])) # task['metrics_suffix']))
        metric_match = list(filter(lambda x: x.startswith(metric_prefix), metrics))
        if metric_match:
            assert rule not in benchmarks
            benchmarks[rule] = {'metrics': defaultdict(dict), 'suffix': task['metrics_suffix']}
            for k, v in parse_metadata(rule).items():
                benchmarks[rule][k] = v

            #print("{} {}".format(metric_prefix, task['metrics_suffix']))
            for metric in metric_match:
                metric_name = extract_metric(metric, metric_prefix, task['metrics_suffix'])
                logging.info("request data for metric {}".format(metric))
                res = cw.get_metric_statistics(Namespace='benchmarkai-metrics-prod',
                                               MetricName=metric,
                                               StartTime=datetime.now() - timedelta(days=1), EndTime=datetime.now(),
                                               Period=86400, Statistics=['Average'])
                print(res)
                points = res['Datapoints']
                if points:
                    if len(points) > 1:
                        logging.warning("More than one datapoint ({}) returned for metric: {}".format(len(points), metric))
                    value = points[0]['Average']
                    #print("metric: {} {i".format(metric_name), value)
                    benchmarks[rule]['metrics'][metric_name] = value
                else:
                    #print("metric: {} N/A".format(metric_name))
                    logging.warning("metric %s : %s without datapoints", rule, metric_name)
                    pass
        else:
            logging.warning("task %s doesn't match metrics", rule)

    return benchmarks

def parse_metadata(metric):
    """Extract the metadata by parsing the metric name.

    Metric names have historically contained metadata about the benchmark. Moving forward, we will
    use a reasonable convention, but for metric names already in use, we will use heuristics to
    parse the metrics.

    Convention: Framework_Benchmark-name_instance-type_metadata_...

    where
      - instance_type is the AWS instance type using the naming convention: p3_16xl, c5_18xl, ....
        Use an empty string if you do not want to specify the instance type.
      - metadata can be any string out of an enumeration, e.g. 'nightly', 'fp32', etc.

    Arguments:
    ---------
        metric:  string
            the metric name

    Returns:
        A dictionary describing the metric metadata.
    """
    # Todo(vishaalk): It's unclear what the source of truth for instance type is.
    whitelist = {
        'Tensorflow_MKL_c5.18xlarge': ('MKL', 'c5_18xl'),
        'Tensorflow_horovod_imagenet_p3.16xlarge_batch_2048': ('Horovod Imagenet', 'p3_16x'),
        'chainer_resnet50_imagenet_sagemaker_ch_docker': ('Resnet50 Imagenet Sagemaker Ch Docker', None),
        'dawnbench_cifar10_gluon': ('Dawnbench Cifar10', None),
        'dawnbench_cifar10_gluon_hybrid': ('Dawnbench Cifar10', None),
        'dawnbench_cifar10_gluon_hybrid_infer': ('Dawnbench Cifar10', None),
        'dawnbench_cifar10_gluon_infer': ('Dawnbench Cifar10', None),
        'dawnbench_cifar10_module': ('Dawnbench Cifar10', None),
        'dawnbench_cifar10_module_infer': ('Dawnbench Cifar10', None),
        'lstm_ptb_imperative_nightly_c4_8x': ('LSTM PTB Imperative', 'c4_8x'),
        'lstm_ptb_imperative_nightly_c5_18x': ('LSTM PTB Imperative', 'c5_18x'),
        'lstm_ptb_imperative_nightly_p2_16x': ('LSTM PTB Imperative', 'p2_16x'),
        'lstm_ptb_imperative_nightly_p3_x': ('LSTM PTB Imperative', 'p3_x'),
        'lstm_ptb_symbolic_nightly_c4_8x': ('LSTM PTB Symbolic', 'c4_8x'),
        'lstm_ptb_symbolic_nightly_c5_18x': ('LSTM PTB Symbolic', 'c5_18x'),
        'lstm_ptb_symbolic_nightly_p2_16x': ('LSTM PTB Symbolic', 'p2_16x'),
        'lstm_ptb_symbolic_nightly_p3_x': ('LSTM PTB Symbolic', 'p3_x'),
        'mms_resnet18_gpu_p3.8x': ('MMS Resnet18', 'p3_8x'),
        'mxnet_resnet50_imagenet_sagemaker_mx_docker': ('Resnet50 Imagenet Sagemaker Mx Docker', None),
        'mxnet_resnet50v1_imagenet_gluon_fp16': ('Resnet50v1 Imagenet', None),
        'mxnet_resnet50v1_imagenet_symbolic_fp16': ('Resnet50v1 Imagenet Symbolic', None),
        'mxnet_resnet50v1_imagenet_symbolic_fp16_p38x': ('Resnet50v1 Imagenet Symbolic', 'p38x'),
        'mxnet_resnet50v1_imagenet_symbolic_fp32': ('Resnet50v1 Imagenet Symbolic', None),
        'mxnet_resnet50v2_imagenet_symbolic_fp16': ('Resnet50v1 Imagenet Symbolic', None),
        'mxnet_resnet50v2_imagenet_symbolic_fp32': ('Resnet50v1 Imagenet Symbolic', None),
        'onnx_mxnet_import_model_inference_test_cpu': ('Onnx Import Model CPU', None),
        'pytorch_resnet50_imagenet_sagemaker_pt_docker': ('Resnet50 Imagenet Sagemaker Pt Docker', None),
        'resnet50_imagenet-480px-256px-q95_p3_16x_fp16_docker': ('Resnet50 Imagenet 480px 256px Q95', 'p3_16x'),
        'tensorflow_resnet50_imagenet_sagemaker_tf_docker': ('Resnet50 Imagenet Sagemaker Tf Docker', None)}

    metadata = {}
    props = metric.lower().split('_')

    # These metric names do not follow a consistent convention, and we will parse them as special
    # cases
    if metric in whitelist.keys():
        metadata['Framework'] = extract_support(props, ['tensorflow', 'mxnet', 'chainer', 'gluon', 'module'])
        metadata['Benchmark'] = whitelist[metric][0]
        metadata['Instance Type'] = whitelist[metric][1]
        metadata['Precision'] = extract_support(props, ['fp16', 'fp32', 'int8'])

    else:
        metadata['Framework'] = props[0]
        metadata['Benchmark'] = props[1]
        metadata['Instance Type'] = props[2]

    if 'infer' in props or 'inference' in props:
        metadata['Type'] = 'Inference'
    else:
        metadata['Type'] = 'Training'

    return metadata


def extract_support(props, support):
    """If an array contains at most one of a set of tokens, return the match.

    Arguments:
    ---------
    props: a list of string
        the properties

    support: a list of string
       the tokens to scan for

    Returns:
        The first support token that is matched in props is returned. If there is more than one
        match, we will log a warning. If there is no match, we return None
    """

    match = None

    for p in props:
        if p in support:
            if match is None:
                match = p
            else:
                logging.warning("Found multiple matches of {} in {}".format(support, props))

    return match
