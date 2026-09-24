"""One direct code-submission attempt. Never pushes or starts a notebook session."""
import json
from pathlib import Path
import time
from kaggle.api.kaggle_api_extended import KaggleApi

OUT=Path(__file__).resolve().parent
REF='yodharinmonplub/biohub-model104-protected-motion-links'


def main():
    receipt=OUT/'hidden_submission_attempt.json'
    if receipt.exists():raise FileExistsError('Inspect previous attempt; do not submit twice')
    result=dict(kernel=REF,kernel_version=1,competition='biohub-cell-tracking-during-development',
        operation='competition_submit_code_only',notebook_push=False,session_created=False,
        attempted_unix=time.time())
    api=KaggleApi();api.authenticate()
    try:
        response=api.competition_submit_code('submission.csv',
            'Model104: original model1 with protected high-confidence motion links; Vast inference verified',
            competition=result['competition'],kernel=REF,kernel_version=1,quiet=True)
        result.update(status='api_returned',response=response.to_dict(response))
    except Exception as error:
        http=getattr(error,'response',None)
        result.update(status='rejected_or_unconfirmed',error_type=type(error).__name__,
            http_status=getattr(http,'status_code',None))
        if http is not None:
            try:
                body=http.json()
                result['error_details']={k:body[k] for k in ['code','message','error','errors'] if k in body}
            except ValueError:
                result['error_details']='Non-JSON response; omitted'
    with receipt.open('x') as f:json.dump(result,f,indent=2,default=str);f.write('\n')
    print(json.dumps(result,indent=2,default=str),flush=True)


if __name__=='__main__':main()
