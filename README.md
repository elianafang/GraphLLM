四个终端执行起来项目MIRAGE

```bash
nohup python3 -m fastchat.serve.controller &
```

```bash
nohup python3 -m fastchat.serve.model_worker     --model-path ~/ONE/MIRAGE/Qwen2-7B-Instruct &
```

```bash
nohup python3 -m fastchat.serve.openai_api_server     --host 0.0.0.0 --port 8000 &
```

```bash
nohup bash run_Qwen.sh > CIL_output_logs/output_$(date +%Y_%m_%d_%H_%M_%S).log 2>&1 &
```

查看fastchat服务是否正常

```
ps ux | grep fastchat
```

查看 run MIRAGE 的情况

```
jobs -l
kill -9 <pid>
```


```markdown
    -----------------
     /     |    \
   home  root   etc
   / \
 whn lfj
```
