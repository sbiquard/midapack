"""Mapmaking with the MAPPRAISER framework."""

from mappraiser.toast_op import operator as mappraiser_op
from sotodlib.toast.workflows.job import workflow_timer


def setup_mapmaker_mappraiser(_parser, operators):
    """Add commandline args and operators for the MAPPRAISER mapmaker.

    Args:
        parser (ArgumentParser):  The parser to update.
        operators (list):  The list of operators to extend.
    """
    operators.append(mappraiser_op.MapMaker(name='mappraiser', enabled=False))


@workflow_timer
def mapmaker_mappraiser(job, otherargs, _runargs, data):
    """Run the MAPPRAISER mapmaker.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.
    """
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.mappraiser.enabled:
        job_ops.mappraiser.output_dir = otherargs.out_dir
        job_ops.mappraiser.pixel_pointing = job.pixels_final
        job_ops.mappraiser.stokes_weights = job.weights_final
        job_ops.mappraiser.apply(data)
