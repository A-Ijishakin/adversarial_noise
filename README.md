Repository for creating adversary examples. 


To use the repo follow these instructions:

1. First clone it with:
    ```
    https://github.com/A-Ijishakin/adversarial_noise.git

    ``` 

2. Then make a virtual environment with:
    ```
    python3.7 -m venv myenv -m venv <name of environemt> 
    ```

3. Then activate the virtual envrionment with:
    ```
    source <name of environment>/bin/activate
    ``` 

4. Install the neccessary packages: 
    ```
    pip install -r requirements.txt 
    ``` 


5. Next run
    ```
    python3 adversary.py --image_path <path to input image> --save_path <path to where the adversarial image should be saved> --target_class <the adversarial class>
    ```

The repo will output the adversarial image at the specified location. 