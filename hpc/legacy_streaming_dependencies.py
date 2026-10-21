"""Complete isolated dependency installation for the frozen Transformers4.33 backend."""
import json
import subprocess

REQUIREMENTS = ['transformers==4.33.3','tokenizers==0.13.3','regex==2024.11.6',
                'numpy==1.26.4','huggingface-hub==0.25.2','safetensors==0.4.5',
                'packaging==24.2','filelock==3.19.1','PyYAML==6.0.3',
                'requests==2.32.5','tqdm==4.67.1']


def install(root):
    subprocess.check_call(['python','-m','pip','install','--target',str(root/'dependencies'),
                           '--upgrade',*REQUIREMENTS])
    # Fresh process sees the isolated target and performs the actual model import.
    probe = "import json,torch,transformers,tokenizers,regex,importlib.metadata as m; from transformers import GPTNeoXConfig,GPTNeoXForCausalLM; print(json.dumps({x:m.version(x) for x in ['torch','transformers','tokenizers','regex','numpy','huggingface-hub']}))"
    output=subprocess.check_output(['python','-c',probe],text=True)
    (root/'dependency-import-check.json').write_text(json.dumps(json.loads(output),indent=2))
    frozen=subprocess.check_output(['python','-m','pip','freeze','--path',str(root/'dependencies')],text=True)
    (root/'dependencies-lock.txt').write_text(frozen)
    print('DEPENDENCY_IMPORTS_PASSED',output.strip(),flush=True)
