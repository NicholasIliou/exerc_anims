import sys
info={}
try:
    import torch
    info['torch']='installed'
    try:
        info['torch_version']=torch.__version__
        info['cuda_available']=torch.cuda.is_available()
        info['cuda_count']=torch.cuda.device_count()
        info['cuda_device_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        info['torch_cuda_error']=str(e)
except Exception as e:
    info['torch']='missing'
    info['torch_err']=str(e)

try:
    import ultralytics
    info['ultralytics']='installed'
    info['ultralytics_version']=ultralytics.__version__
except Exception as e:
    info['ultralytics']='missing'
    info['ultralytics_err']=str(e)

try:
    import mediapipe as mp
    info['mediapipe']='installed'
    info['mediapipe_version']=mp.__version__
except Exception as e:
    info['mediapipe']='missing'
    info['mediapipe_err']=str(e)

print('DIAGNOSTIC')
for k,v in info.items():
    print(f"{k}: {v}")
