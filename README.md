## Docker

### Build
```sh
docker build -t the_bass/pytorch .
```

### Bash
```sh
docker run -it --rm \
    -w /app \
    -v `pwd`:/app \
    the_bass/pytorch \
    bash
```

### Jupyter Notebook
```sh
docker run -it --rm \
    -w /app \
    -v `pwd`:/app \
    -p 8888:8888 \
    --name jupyter \
    the_bass/pytorch \
    jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root
```
```sh
docker exec -it jupyter bash
```


python -m unittest discover tests
