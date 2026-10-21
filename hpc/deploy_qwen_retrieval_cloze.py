"""Submit the separately labelled completion-format follow-up."""
import deploy_qwen_long_retrieval as deploy

deploy.NAME='qwen05-long-retrieval-cloze-20261016-r1'
deploy.LOCAL=deploy.ROOT/'artifacts'/deploy.NAME
deploy.REMOTE='/cluster/users/alex116u1/data/'+deploy.NAME
deploy.FOLLOWUP=True

if __name__=='__main__':deploy.main()
