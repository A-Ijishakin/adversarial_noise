Repository for creating adversarial examples. 

This repository requires python >= 3.7.2. Please select an adversarial class from the classes.txt file. Note that the script will expect the class name to have the exact formatting from the classes.txt file. 

To use the repo follow these instructions:

1. First clone it with:
    ```
    https://github.com/A-Ijishakin/adversarial_noise.git

    ``` 

2. Then cd into it
   ```
   cd adversarial_noise  
   ```

3. Then make a virtual environment with:
    ```
    python3 -m venv <name of environemt> 
    ```

4. Then activate the virtual envrionment with:
    ```
    source <name of environment>/bin/activate
    ``` 

5. Install the neccessary packages: 
    ```
    pip install -r requirements.txt 
    ``` 


5. Next run
    ```
    python3 produce_adversarial_example.py --image_path <path to input image> --save_path <path to where the adversarial image should be saved> --target_class <the adversarial class>
    ```

The repo will output the adversarial image at the specified location. 



