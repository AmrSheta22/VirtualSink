"""Collect the separately labelled completion-format evaluation."""
import collect_qwen_long_retrieval as collect
from deploy_qwen_long_retrieval import ROOT

collect.LOCAL=ROOT/'artifacts/qwen05-long-retrieval-cloze-20261016-r1'
collect.REMOTE='/cluster/users/alex116u1/data/qwen05-long-retrieval-cloze-20261016-r1'

if __name__=='__main__':collect.main()
