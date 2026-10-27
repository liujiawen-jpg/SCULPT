"""Exercise the scientific path without using private data or a GPU."""
from cosit.smoke import run_smoke


def test_actual_j0_end_to_end(tmp_path):
    report=run_smoke(tmp_path/'synthetic')
    assert report['status']=='passed'
    assert report['conditions']==12 and report['genes']==48
    assert 'PDS_delta' in report['response_metrics']
    assert 'DEG_AP' in report['response_metrics']
    assert 'MMD2' in report['distribution_metrics']
    assert 'centered_joint_SW1_normalized' in report['distribution_metrics']
    assert report['formal_experiment'] is False
