from adversary import Adversary 
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parse the input arguments 
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--target_class', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args() 
    
    #test the visualise image and noise function we now have an example adversarial image and noise which can be viewed in the output folder
    Adversary().produce_adversarial_image(image_path=args.image_path, save_path = args.save_path, 
                                          target_class = args.target_class)  