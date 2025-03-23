# Transferencia de escena acústica utilizando codecs de audio neuronales

1. Instalar [uv](https://docs.astral.sh/uv/)
2. Recrear el entorno virtual con el comando 
    ```shell
    uv venv .venv && uv pip sync
    ```
3. Bajar dataset de entrenamiento raw
    
    a. Con  bash: 
    ```shell
    chmod +x ./data/EARS/ears_dataset/download_ears.sh
    sh ./data/EARS/ears_dataset/download_ears.sh
    ``` 
    b. Con python corriendo `./data/EARS/ears_dataset/download_ears.py`.
4. Bajar las rirs con el comando (hay que activar previamente el env para que funcionen ciertos CLI commands)
    ```shell
    chmod +x ./data/EARS/rir_dataset/download_ears_reverb.sh
    sh ./data/EARS/rir_dataset/download_ears_reverb.sh
    ```
> Ciertas partes van a fallar, quizá hay que bajar algo manualmente, acomodarlo para que funcione y luego correr el script de nuevo, etc