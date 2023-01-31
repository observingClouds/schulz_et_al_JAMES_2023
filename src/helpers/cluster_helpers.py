def setup_cluster(cluster="slurmcluster", verbose="debug", jupyterhub=True):
    """Wrapper to quickly setup a dask distributed cluster.

    Input
    -----
    cluster : str
       Type of cluster that shall be started.
       Currently a slurmcluster and a local cluster are supported.
    """

    import dask
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import SLURMCluster

    print(dask.__version__)
    if jupyterhub:
        dask.config.config.get("distributed").get("dashboard").update(
            {"link": "{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status"}
        )

    if cluster == "slurmcluster":
        cluster = SLURMCluster(
            project="mh0010",
            cores=128,
            interface="ib0",
            walltime="00:20:00",
            extra=[
                "--lifetime",
                "15m",
                "--lifetime-stagger",
                "4m",
            ],  # When using cluster.adapt()
            silence_logs=verbose,
            memory="256GB",
            queue="compute",
            job_extra=[
                "--output=/work/mh0010/m300408/LOG.DASK.%j.o",
                "--error=/work/mh0010/m300408/LOG.DASK.%j.o",
                "--exclusive",
            ],
            # scheduler_file='/scratch/m/m300408/scheduler.json',
            asynchronous=0,
        )
        client = Client(cluster)
    else:
        with dask.config.set({"distributed.scheduler.worker-saturation": "1.0"}):
            cluster_local = LocalCluster(
                interface="ib0",
                silence_logs=verbose,
                n_workers=16,
                threads_per_worker=4,
            )  # , env={"MALLOC_TRIM_THRESHOLD_":-1})
            client = Client(cluster_local)
    client.amm.start()
    return client
