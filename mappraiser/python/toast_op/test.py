import os
from pathlib import Path

import numpy as np
import toast.ops as ops
from mpi4py.MPI import Comm, Intracomm
from toast.observation import default_values as defaults
from toast.tests._helpers import (
    close_data,
    create_fake_sky,
    create_satellite_data,
    fake_flags,
)
from toast.tests.mpi import MPITestCase
from toast.vis import set_matplotlib_backend


def create_outdir(
    mpicomm: Comm | Intracomm | None, subdir: str | os.PathLike | None = None
) -> Path:
    """Create the top level output directory and per-test subdir.

    Args:
        mpicomm: the MPI communicator (or None).
        subdir: the subdirectory for this test.

    Returns:
        Full path to the test subdir if specified, else the top dir.
    """
    cwd = Path.cwd()
    testdir = cwd / 'mappraiser_test_output'
    retdir = testdir
    if subdir is not None:
        retdir = testdir / subdir
    if (mpicomm is None) or (mpicomm.rank == 0):
        testdir.mkdir(exist_ok=True, parents=True)
        retdir.mkdir(exist_ok=True, parents=True)
    if mpicomm is not None:
        mpicomm.barrier()
    return retdir


class InterfaceTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def runTest(self):
        try:
            from .operator import MapMaker
        except ImportError:
            print('Mappraiser not available, skipping tests')
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
            create_dist='pixel_dist',
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode='IQU',
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, 'pixel_dist', 'fake_map')

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key='fake_map',
        )
        scanner.apply(data)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            xdata = ob.shared['times'].data
            ydata = ob.detdata['signal'][det]

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect='auto')
            ax.plot(
                xdata,
                ydata,
                marker='o',
                c='red',
                label=f'{ob.name}, {det}',
            )
            # cur_ylim = ax.get_ylim()
            # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
            ax.legend(loc=1)
            plt.title('Sky Signal')
            savefile = self.outdir / f'signal_sky_{ob.name}_{det}.pdf'
            plt.savefig(savefile)
            plt.close()

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        print(
            f"psd unit = {data.obs[0]['noise_model'].psd(data.obs[0].local_detectors[0]).unit}",
            flush=True,
        )

        # Simulate noise from this model
        sim_noise = ops.SimNoise(det_data='noise')
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            xdata = ob.shared['times'].data
            ydata = ob.detdata['signal'][det]

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect='auto')
            ax.plot(
                xdata,
                ydata,
                marker='o',
                c='red',
                label=f'{ob.name}, {det}',
            )
            # cur_ylim = ax.get_ylim()
            # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
            ax.legend(loc=1)
            plt.title('Sky + Noise Signal')
            savefile = self.outdir / f'signal_sky-noise_{ob.name}_{det}.pdf'
            plt.savefig(savefile)
            plt.close()

        # Compute timestream rms

        rms = dict()
        for ob in data.obs:
            rms[ob.name] = dict()
            for det in ob.local_detectors:
                flags = np.array(ob.shared[defaults.shared_flags])
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                # Add an offset to the data
                ob.detdata[defaults.det_data][det] += 500.0
                rms[ob.name][det] = np.std(ob.detdata[defaults.det_data][det][good])

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            xdata = ob.shared['times'].data
            ydata = ob.detdata['signal'][det]

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect='auto')
            ax.plot(
                xdata,
                ydata,
                marker='o',
                c='red',
                label=f'{ob.name}, {det}',
            )
            # cur_ylim = ax.get_ylim()
            # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
            ax.legend(loc=1)
            plt.title('Sky + Noise + Offset Signal')
            savefile = self.outdir / f'signal_sky-noise-offset_{ob.name}_{det}.pdf'
            plt.savefig(savefile)
            plt.close()

        # Run mappraiser on this

        mapmaker = MapMaker(
            output_dir=str(self.outdir),
            lagmax=16,
            maxiter=50,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_name='noise',
            purge_det_data=False,
            mem_report=True,
        )
        mapmaker.apply(data)

        close_data(data)
