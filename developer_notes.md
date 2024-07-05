# Developer notes

## Software environment

Any UNSEEN workflow requires command line programs from the UNSEEN package at https://github.com/AusClimateService/unseen.

A copy of that repo has been cloned to `/g/data/xv83/unseen-projects/unseen`
and then installed into the Python virtual environment at `/g/data/xv83/unseen-projects/unseen_venv`.
The creation of that virtual environment followed the
[CLEX instructions](https://climate-cms.org/posts/2024-06-05-mixing-python-envs.html):

```
module use /g/data/hh5/public/modules
module load conda/analysis3
cd /g/data/xv83/unseen-projects/
python3 -m venv unseen_venv --system-site-packages
source /g/data/xv83/unseen-projects/unseen_venv/bin/activate
cd /g/data/xv83/unseen-projects/unseen
pip install --no-deps -e .
pip install xstatstests
pip install papermill
```



