Repository for creating adversary examples. 

It requires python >= 3.7 and the pacakges in the requirements file


To use the repo follow these instructions:

1. First clone it with:
    ```
    https://github.com/A-Ijishakin/adversarial_noise.git

    ```
2. Next run
    ```
    python3 adversary.py --image_path <path to input image> --save_path <path to where the adversarial image should be saved> --target_class <the adversarial class>
    ```

The repo will output the adversarial image at the specified location 