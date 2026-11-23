from __future__ import annotations
import os, uuid, time, logging, threading
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parents[1]; UPLOADS=ROOT/'uploads'; RESULTS=ROOT/'results'
UPLOADS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
MAX_BYTES=int(os.getenv('MAX_UPLOAD_MB','500'))*1024*1024
MODEL_NAME=os.getenv('YOLO_MODEL','yolo11n.pt')
MODEL_SETTING=Path(MODEL_NAME)
MODEL_PATH=MODEL_SETTING if MODEL_SETTING.is_absolute() else ROOT/MODEL_SETTING
MODEL_SOURCE=str(MODEL_PATH) if MODEL_PATH.is_file() else MODEL_NAME
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO')); log=logging.getLogger('visionpark')
app=FastAPI(title='VisionPark AI API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(','),allow_methods=['*'],allow_headers=['*'])
jobs:dict[str,dict[str,Any]]={}; model=None; model_lock=threading.Lock(); job_lock=threading.Lock()
VEHICLES={2:'car',3:'motorcycle',5:'bus',7:'truck'}

def estimate_parking_grid(image:np.ndarray,boxes:np.ndarray)->list[list[list[float]]]:
    """Estimate visible stalls from detected parked-vehicle rows and repeated spacing.

    Only gaps between visible vehicles are inferred. Empty end-of-row stalls are
    not guessed, because there is no reliable boundary evidence for them.
    """
    h,w=image.shape[:2]
    if len(boxes)<2: return []
    centers=np.column_stack(((boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2))
    centers=centers[np.argsort(centers[:,1])]
    rows=[]
    for x,y in centers:
        if not rows or y-float(np.median([p[1] for p in rows[-1]]))>h*.085: rows.append([])
        rows[-1].append((float(x),float(y)))
    rows=[sorted(row) for row in rows if len(row)>=2]
    if not rows: return []
    row_y=[float(np.median([p[1] for p in row])) for row in rows]
    gaps=np.diff(row_y); typical=float(np.median(gaps)) if len(gaps) else h*.17
    row_height=max(h*.075,min(h*.22,typical*.72))
    polygons=[]
    for row in rows:
        xs=np.array([p[0] for p in row]); ys=np.array([p[1] for p in row])
        diffs=np.diff(xs); plausible=diffs[(diffs>=w*.07)&(diffs<=w*.23)]
        if not len(plausible): continue
        pitch=float(np.median(plausible))
        if pitch<w*.07: continue
        centers_x=[float(xs[0])]
        for left,right in zip(xs,xs[1:]):
            steps=max(1,min(3,int(round((right-left)/pitch))))
            # Large gaps are only expanded when the repeated spacing supports it.
            if (right-left)>pitch*2.45: steps=1
            centers_x.extend(float(left+(right-left)*k/steps) for k in range(1,steps+1))
        cy=float(np.median(ys)); slot_w=pitch*.84; slot_h=row_height
        for cx in centers_x:
            left=max(0,cx-slot_w/2); right=min(w,cx+slot_w/2)
            top=max(0,cy-slot_h/2); bottom=min(h,cy+slot_h/2)
            if right-left<w*.035 or bottom-top<h*.04: continue
            polygons.append([[left/w,top/h],[right/w,top/h],[right/w,bottom/h],[left/w,bottom/h]])
    return polygons if 2<=len(polygons)<=80 else []

def estimate_slots_from_markings(image:np.ndarray)->list[list[list[float]]]:
    """Best-effort stall proposals from long, visible painted line segments."""
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),45,135)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(35,min(h,w)//14),
                          minLineLength=max(28,int(min(h,w)*.055)),maxLineGap=max(12,int(min(h,w)*.035)))
    if lines is None: return []
    vertical=[]; horizontal=[]
    for raw in lines[:,0,:]:
        x1,y1,x2,y2=map(int,raw); dx=x2-x1; dy=y2-y1; length=(dx*dx+dy*dy)**.5
        if length<min(w,h)*.055: continue
        angle=abs(np.degrees(np.arctan2(dy,dx)))
        if angle<12 or angle>168: horizontal.append((x1+x2)/2)
        elif 76<angle<104: vertical.append((y1+y2)/2)
    def merge(values,scale):
        values=sorted(values); groups=[]
        for value in values:
            if not groups or value-float(np.mean(groups[-1]))>scale: groups.append([value])
            else: groups[-1].append(value)
        return [float(np.mean(g)) for g in groups if len(g)>=1]
    xs=merge(horizontal,max(5,w*.012)); ys=merge(vertical,max(5,h*.012))
    slots=[]
    for y1,y2 in zip(ys,ys[1:]):
        sh=y2-y1
        if not h*.07<=sh<=h*.38: continue
        for x1,x2 in zip(xs,xs[1:]):
            sw=x2-x1; ratio=sw/max(1,sh)
            if not w*.035<=sw<=w*.28 or not .55<=ratio<=3.8: continue
            slots.append([[x1/w,y1/h],[x2/w,y1/h],[x2/w,y2/h],[x1/w,y2/h]])
    return slots[:80]

def get_model():
    global model
    with model_lock:
        if model is None:
            try:
                from ultralytics import YOLO
                candidate=YOLO(MODEL_SOURCE)
                # Force actual model initialization; no silent mocked fallback.
                model=candidate
                log.info('Loaded model %s',MODEL_SOURCE)
            except Exception as exc: raise RuntimeError(f'Could not load {MODEL_NAME}: {exc}') from exc
    return model

async def save_upload(file:UploadFile,allowed:set[str])->Path:
    ext=Path(file.filename or '').suffix.lower()
    if ext not in allowed: raise HTTPException(415,f'Unsupported file extension: {ext or "none"}')
    path=UPLOADS/f'{uuid.uuid4().hex}{ext}'; size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_BYTES: raise HTTPException(413,'Upload exceeds configured size limit')
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True); raise
    if size==0: path.unlink(missing_ok=True); raise HTTPException(400,'Uploaded file is empty')
    return path

def public_job(j):
    return {k:v for k,v in j.items() if k not in {'source_path','output_path','csv_path'}}

@app.get('/api/health')
def health():
    return {'status':'ok','model':MODEL_NAME,'model_loaded':model is not None,'device':'cuda-if-available-else-cpu'}

@app.post('/api/vehicle/process',status_code=202)
async def vehicle_process(background_tasks:BackgroundTasks,file:UploadFile=File(...),line_y:float=0.58):
    source=await save_upload(file,{'.mp4','.avi','.mov','.mkv'})
    jid=uuid.uuid4().hex
    jobs[jid]={'job_id':jid,'kind':'vehicle','status':'queued','progress':0,'filename':file.filename,'created_at':time.time(),'source_path':str(source)}
    background_tasks.add_task(process_video,jid,source,line_y)
    return {'job_id':jid,'status':'queued'}

def process_video(jid:str,source:Path,line_y:float):
    j=jobs[jid]; cap=cv2.VideoCapture(str(source)); writer=None; started=time.perf_counter(); counts={n:0 for n in VEHICLES.values()}; counted=set(); last={}; total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    if not cap.isOpened(): j.update(status='error',error='OpenCV could not decode this video.'); source.unlink(missing_ok=True); return
    try:
        yolo=get_model(); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); output=RESULTS/f'{jid}.mp4'; csv=RESULTS/f'{jid}.csv'
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened(): raise RuntimeError('Could not create processed video output.')
        j.update(status='running',output_path=str(output),csv_path=str(csv),progress=1)
        device=0 if __import__('torch').cuda.is_available() else 'cpu'
        frame_idx=0; preview_frame=None; preview_score=-1
        while True:
            ok,frame=cap.read()
            if not ok: break
            result=yolo.track(frame, persist=True, tracker='bytetrack.yaml', classes=list(VEHICLES), conf=0.25, imgsz=640, device=device, verbose=False)[0]
            cv2.line(frame,(0,int(height*line_y)),(width,int(height*line_y)),(255,190,65),2)
            if result.boxes is not None:
                boxes=result.boxes.xyxy.cpu().numpy(); clss=result.boxes.cls.cpu().numpy().astype(int); ids=result.boxes.id
                ids=ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes),-1)
                for box,cls,tid in zip(boxes,clss,ids):
                    if cls not in VEHICLES: continue
                    x1,y1,x2,y2=map(int,box); cy=(y1+y2)/2; key=int(tid)
                    label=VEHICLES[cls]; cv2.rectangle(frame,(x1,y1),(x2,y2),(61,224,173),2); cv2.putText(frame,f'{label.title()}  #{key}',(x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(61,224,173),2)
                    if key>=0 and key in last and (last[key]-height*line_y)*(cy-height*line_y)<0 and key not in counted:
                        counts[label]+=1; counted.add(key)
                    if key>=0: last[key]=cy
            cv2.rectangle(frame,(16,16),(255,107),(10,15,24),-1)
            cv2.putText(frame,f'CROSSED  {sum(counts.values())}',(28,50),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2)
            cv2.putText(frame,f'CAR {counts["car"]}  MOTO {counts["motorcycle"]}  BUS {counts["bus"]}  TRUCK {counts["truck"]}',(28,83),cv2.FONT_HERSHEY_SIMPLEX,.52,(73,211,192),2)
            score=len(result.boxes) if result.boxes is not None else 0
            if score>=preview_score: preview_score=score; preview_frame=frame.copy()
            writer.write(frame); frame_idx+=1
            if total_frames: j['progress']=min(99,round(frame_idx/total_frames*100))
        elapsed=time.perf_counter()-started
        preview=RESULTS/f'{jid}-preview.jpg'
        if preview_frame is not None: cv2.imwrite(str(preview),preview_frame)
        pd.DataFrame([{'category':k,'crossing_count':v} for k,v in counts.items()]).to_csv(csv,index=False)
        j.update(status='completed',progress=100,counts=counts,total=sum(counts.values()),frames=frame_idx,processing_seconds=round(elapsed,2),average_fps=round(frame_idx/elapsed,2) if elapsed else 0,preview_url=f'/api/files/{jid}/preview',output_url=f'/api/files/{jid}/video',csv_url=f'/api/files/{jid}/csv')
    except Exception as exc:
        log.exception('Video processing failed'); j.update(status='error',error=str(exc) if 'Could not load' in str(exc) else f'Video processing failed: {exc}')
    finally:
        cap.release()
        if writer: writer.release()
        source.unlink(missing_ok=True)

class SlotRequest(BaseModel): polygons:list[list[list[float]]]

@app.post('/api/parking/process',status_code=202)
async def parking_process(background_tasks:BackgroundTasks,file:UploadFile=File(...),polygons:str=Form('[]')):
    import json
    source=await save_upload(file,{'.jpg','.jpeg','.png','.webp'})
    try: slots=json.loads(polygons)
    except Exception: source.unlink(missing_ok=True); raise HTTPException(422,'polygons must be valid JSON')
    if not isinstance(slots,list): source.unlink(missing_ok=True); raise HTTPException(422,'polygons must be a JSON list')
    for poly in slots:
        pts=np.array(poly,dtype=np.float32)
        if pts.ndim!=2 or pts.shape[0]<3 or pts.shape[1]!=2 or np.any(pts<0) or np.any(pts>1):
            source.unlink(missing_ok=True); raise HTTPException(422,'Each slot must be a polygon with 3+ normalized [0,1] points')
    jid=uuid.uuid4().hex
    jobs[jid]={'job_id':jid,'kind':'parking','status':'queued','progress':0,'filename':file.filename,'created_at':time.time(),'source_path':str(source)}
    background_tasks.add_task(process_parking,jid,source,slots)
    return {'job_id':jid,'kind':'parking','status':'queued','progress':0}

def process_parking(jid:str,source:Path,slots:list):
    j=jobs[jid]; started=time.perf_counter()
    try:
        image=cv2.imread(str(source))
        if image is None: raise RuntimeError('Image could not be decoded')
        j.update(status='running',progress=5)
        yolo=get_model(); device=0 if __import__('torch').cuda.is_available() else 'cpu'
        result=yolo.predict(image,classes=[2,5,7],conf=.35,imgsz=640,agnostic_nms=True,device=device,verbose=False)[0]
        h,w=image.shape[:2]; boxes=result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0,4))
        vehicle_boxes=[[float(x1/w),float(y1/h),float(x2/w),float(y2/h)] for x1,y1,x2,y2 in boxes]
        auto_configured=False
        if not slots:
            slots=estimate_parking_grid(image,boxes)
            if not slots: slots=estimate_slots_from_markings(image)
            auto_configured=bool(slots)
        occupancy=[]
        for idx,poly in enumerate(slots):
            pts=np.array(poly,dtype=np.float32); px=(pts*np.array([w,h])).astype(np.int32)
            mask=np.zeros((h,w),np.uint8); cv2.fillPoly(mask,[px],1); occupied=False
            for x1,y1,x2,y2 in boxes:
                bx1=max(0,int(np.floor(x1))); by1=max(0,int(np.floor(y1))); bx2=min(w,int(np.ceil(x2))); by2=min(h,int(np.ceil(y2)))
                if bx2<=bx1 or by2<=by1: continue
                crop=mask[by1:by2,bx1:bx2]
                center_inside=bool(mask[min(h-1,max(0,int((y1+y2)/2))),min(w-1,max(0,int((x1+x2)/2)))])
                if center_inside or crop.sum()/((bx2-bx1)*(by2-by1))>=.18: occupied=True; break
            occupancy.append({'slot':idx+1,'occupied':occupied})
            color=(47,80,238) if occupied else (63,213,151); overlay=image.copy()
            cv2.fillPoly(overlay,[px],color); cv2.addWeighted(overlay,.23,image,.77,0,image); cv2.polylines(image,[px],True,color,3)
            cx,cy=np.mean(px,axis=0).astype(int); cv2.putText(image,str(idx+1),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        for x1,y1,x2,y2 in boxes:
            cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(50,80,240),3)
            cv2.putText(image,'Vehicle',(int(x1),max(20,int(y1)-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,80,240),2)
        j['progress']=85
        out=RESULTS/f'{jid}.jpg'; csv=RESULTS/f'{jid}.csv'; cv2.imwrite(str(out),image)
        pd.DataFrame(occupancy).to_csv(csv,index=False); occupied=sum(x['occupied'] for x in occupancy)
        j.update(status='completed',progress=100,auto_configured=auto_configured,needs_manual_configuration=False,
            message=None if slots else 'No clear parking bay markings or repeated vehicle pattern were visible, so this image did not provide a reliable slot total. Try a wider, clearer parking-lot image.',
            total_slots=len(occupancy),occupied=occupied,available=len(occupancy)-occupied,
            occupancy_percent=round(occupied/len(occupancy)*100,1) if occupancy else 0,slots=occupancy,polygons=slots,
            detected_vehicles=len(boxes),vehicle_boxes=vehicle_boxes,image_url=f'/api/files/{jid}/image',csv_url=f'/api/files/{jid}/csv',
            processing_seconds=round(time.perf_counter()-started,2))
    except Exception as exc:
        log.exception('Parking analysis failed'); j.update(status='error',progress=100,error=f'Parking analysis failed: {exc}')
    finally:
        source.unlink(missing_ok=True)

@app.get('/api/jobs/{jid}')
def job_status(jid:str):
    if jid not in jobs: raise HTTPException(404,'Job not found or expired')
    return public_job(jobs[jid])

@app.get('/api/files/{jid}/{kind}')
def get_file(jid:str,kind:str):
    if kind=='video': p=RESULTS/f'{jid}.mp4'; media='video/mp4'
    elif kind=='image': p=RESULTS/f'{jid}.jpg'; media='image/jpeg'
    elif kind=='csv': p=RESULTS/f'{jid}.csv'; media='text/csv'
    elif kind=='preview': p=RESULTS/f'{jid}-preview.jpg'; media='image/jpeg'
    else: raise HTTPException(404,'File not found')
    if not p.is_file(): raise HTTPException(404,'File not found')
    return FileResponse(p,media_type=media,filename=p.name)

@app.get('/api/history')
def history(): return {'items':[public_job(j) for j in jobs.values() if j['status']=='completed'][-50:]}
