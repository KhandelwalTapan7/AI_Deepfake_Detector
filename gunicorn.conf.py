# gunicorn.conf.py
timeout = 120          # workers get 120s per request (default is 30s — too short for CPU inference)
workers = 1            # single worker saves RAM on free tier
threads = 2
worker_class = "sync"
bind = "0.0.0.0:10000"
