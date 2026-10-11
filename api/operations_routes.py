"""Protected operator session/status/job adapters."""
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.security import APIKeyCookie, HTTPBearer, HTTPAuthorizationCredentials
from application.web_schemas import JobRequest,JobRecord,JobPage,OperatorSession,OperatorStatus

operator_bearer=HTTPBearer(auto_error=False,scheme_name='OperatorBearer',description='Configured private operator token, distinct from the optional shared read token.')
local_cookie=APIKeyCookie(name='stockpro_operator_session',auto_error=False,scheme_name='LocalOperatorCookie',description='Explicit local loopback session; exact configured Origin required on every request.')

def require_operator(request:Request,bearer:HTTPAuthorizationCredentials|None=Depends(operator_bearer),cookie:str|None=Depends(local_cookie)):
    return request.app.state.operator_auth.require(request)

def operations_router():
    router=APIRouter(prefix='/api/v1/operations',tags=['operator'])
    @router.post('/session',response_model=OperatorSession)
    def session(request:Request,response:Response):return request.app.state.operator_auth.provision(request,response)
    @router.get('/session',response_model=OperatorSession,dependencies=[Depends(require_operator)])
    def current_session(request:Request):return request.app.state.operator_auth.require(request)
    @router.delete('/session',response_model=OperatorSession,dependencies=[Depends(require_operator)])
    def logout(request:Request,response:Response):return request.app.state.operator_auth.revoke(request,response)
    @router.get('/status',response_model=OperatorStatus,dependencies=[Depends(require_operator)])
    def status(request:Request):return request.app.state.operator_jobs.status()
    @router.post('/jobs',response_model=JobRecord,status_code=202,dependencies=[Depends(require_operator)])
    def enqueue(body:JobRequest,request:Request):return request.app.state.operator_jobs.enqueue(body.job_type,body.origin_session.isoformat() if body.origin_session else None)
    @router.get('/jobs',response_model=JobPage,dependencies=[Depends(require_operator)])
    def jobs(request:Request,offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=100)):return request.app.state.operator_jobs.page(offset,limit)
    @router.get('/jobs/{run_id}',response_model=JobRecord,dependencies=[Depends(require_operator)])
    def job(run_id:str,request:Request):return request.app.state.operator_jobs.get(run_id)
    return router
