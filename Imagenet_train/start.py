import subprocess
import os
print('////////////////////////////////////////////////////////////////////')
print(os.path.abspath(os.curdir))
script_path = '/home/work/user-job-dir/imagenet_distill_interpolate/console/job_cneast3_v1/job.sh'
#script_path = '/home/work/user-job-dir/once-for-all-dist/connect_multistage_training.sh'
#script_path = '/home/ma-user/modelarts/user-job-dir/once-for-all-dist/connect_multistage_training.sh'
sp = subprocess.Popen(['bash', script_path])
sp.wait()