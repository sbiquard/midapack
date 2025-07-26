"""Mapmaking with the MAPPRAISER framework."""

from sotodlib.toast.workflows.job import workflow_timer

from mappraiser.toast_op import operator as mappraiser_op


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

        if getattr(otherargs, 'scramble_after', None):
            # We want to scramble gains after estimating noise
            job_ops.mappraiser.scrambling = job_ops.gainscrambler

        # Handle "single det" mode
        # Works speficically for sotodlib workflows where detectors come in A/B pairs
        dets = None
        if otherargs.single_det:
            dets = [d for d in data.all_local_detectors() if d.endswith('A')]
        job_ops.mappraiser.apply(data, detectors=dets)
