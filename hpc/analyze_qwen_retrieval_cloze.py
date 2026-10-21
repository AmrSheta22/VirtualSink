"""Analyze completion-format probes without replacing the Q/A analysis."""
import analyze_qwen_long_retrieval as analysis
from deploy_qwen_long_retrieval import ROOT

analysis.LOCAL=ROOT/'artifacts/qwen05-long-retrieval-cloze-20261016-r1'

if __name__=='__main__':analysis.main()
