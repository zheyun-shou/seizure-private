import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from timescoring.annotations import Annotation
from timescoring import scoring, visualization
from timescoring.scoring import SampleScoring, EventScoring

def pretty_print_matrix(matrix):
    """Pretty print the np.array matrix with proper formatting and line breaks."""
    # Convert to string with proper formatting
    matrix_str = np.array2string(matrix, separator=', ', precision=2, suppress_small=True)
    
    # Add line breaks after each row
    matrix_str = matrix_str.replace('\n', '\n\t')

    print(matrix_str)

def pretty_print_dict(dict):
    for key, value in dict.items():
        print(f"{key}: {value}")

def analyze_classification(y_pred, y_true):
    """Analyze the classification performance."""
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat_norm = confusion_matrix(y_true, y_pred, normalize='true')
    summary = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'conf_mat': conf_mat,
        'conf_mat_norm': conf_mat_norm
    }
    print(f"\nEpoch-wise model accuracy on test set: {accuracy:.2f}")
    print(f"\nEpoch-wise model f1 score on test set: {f1:.2f}")
    print(f"\nEpoch-wise model precision on test set: {precision:.2f}")
    print(f"\nEpoch-wise model recall on test set: {recall:.2f}")
    print(f"\nEpoch-wise model confusion matrix on test set:")
    pretty_print_matrix(conf_mat)
    print(f"\nEpoch-wise model normalized confusion matrix on test set:")
    pretty_print_matrix(conf_mat_norm)
    return summary

def timescoring_analysis(ref_events, hyp_events, fs, n_samples, recording_ids, config):
    """Analyze the classification performance using timescoring."""
    ref = Annotation(ref_events, fs, n_samples)
    hyp = Annotation(hyp_events, fs, n_samples)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=100000,
        minDurationBetweenEvents=90)
    sample_scores = SampleScoring(ref, hyp)
    event_scores = EventScoring(ref, hyp, param)

    # visualize the results
    if config['generate_timescoring_plots']:
        fig, ax = plt.subplots(2, 1, figsize=(8, 4.8))
        figSamples = visualization.plotSampleScoring(ref, hyp, ax=ax[0])
        figEvents = visualization.plotEventScoring(ref, hyp, param, ax=ax[1])
        ss_path = "{}/{}_{}/{}_sub-{}_ses-{}_{}_run-{}.png".format(
            config['result_dir'], config['model_name'], config['dataset'], 
            "sample_scoring", recording_ids['subject'], recording_ids['session'], recording_ids['task'], recording_ids['run'])
        es_path = "{}/{}_{}/{}_sub-{}_ses-{}_{}_run-{}.png".format(
            config['result_dir'], config['model_name'], config['dataset'], 
            "event_scoring", recording_ids['subject'], recording_ids['session'], recording_ids['task'], recording_ids['run'])
        os.makedirs(os.path.dirname(ss_path), exist_ok=True)
        os.makedirs(os.path.dirname(es_path), exist_ok=True)
        ax[0].figure.savefig(ss_path, bbox_inches='tight')
        ax[1].figure.savefig(es_path, bbox_inches='tight')
        plt.close(figSamples)
        plt.close(figEvents)
    
    summary = {
        "subject_id": recording_ids['subject'],
        "session_id": recording_ids['session'],
        "task_id": recording_ids['task'],
        "run_id": recording_ids['run'],
        'sample_sensitivity': sample_scores.sensitivity,
        'sample_precision': sample_scores.precision,
        'sample_f1': sample_scores.f1,
        'sample_fpRate': sample_scores.fpRate,
        'event_sensitivity': event_scores.sensitivity,
        'event_precision': event_scores.precision,
        'event_f1': event_scores.f1,
        'event_fpRate': event_scores.fpRate,
    }

    # print(f"\nTimescoring analysis for recording {recording_ids}:")
    # pretty_print_dict(summary)

    return summary
