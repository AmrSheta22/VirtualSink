"""CPU Slurm repair: reuse verified assets, install complete dependencies in a fresh root."""
import hashlib,json,os
from pathlib import Path
from legacy_streaming_dependencies import install


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(os.environ['REPLICATION_ROOT'])
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['data_sha256'].items():
        assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()==expected,name
    install(root)
    print('ASSETS_VERIFIED_AND_DEPENDENCIES_READY',flush=True)


if __name__=='__main__':main()
