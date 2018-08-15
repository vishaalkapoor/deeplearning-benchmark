#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities to fetch metrics from Benchmark.AI.

This code was modified from code originally written by Pedro Larroy"""

import boto3
import json
import logging
import pickle

from collections import defaultdict
from datetime import datetime,timedelta


events = boto3.client('events')
cw = boto3.client('cloudwatch')

def get_metrics():
    p = cw.get_paginator('list_metrics')
    i = p.paginate(Namespace='benchmarkai-metrics-prod')
    res = i.build_full_result()
    return list(map(lambda x: x['MetricName'], res['Metrics']))


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
                        logging.warn("More than one datapoint ({}) returned for metric: {}".format(len(points), metric))
                    value = points[0]['Average']
                    #print("metric: {} {i".format(metric_name), value)
                    benchmarks[rule]['metrics'][metric_name] = value
                else:
                    #print("metric: {} N/A".format(metric_name))
                    logging.warn("metric %s : %s without datapoints", rule, metric_name)
                    pass
        else:
            logging.warning("task %s doesn't match metrics", rule)
    return benchmarks

