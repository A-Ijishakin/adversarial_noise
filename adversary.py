import torch 
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights  
import torch.nn as nn
from torch.nn import functional as F 
import matplotlib.pyplot as plt
import argparse 

class Adversary:
    """ 
    Adversary: 
        A class that will be used to generate adversarial examples, the class will take an image and a target class and
        produce an adversarial example that will be classified as the target class by a pretrained ResNet50 model. 
    
    """
    def __init__(self):
        """ 
        Initialises the Adversary class.
        
        """
        
        #set the device to cuda if available, otherwise cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #instantite the model weights
        self.weights = ResNet50_Weights.DEFAULT
        
        #load in the model 
        self.model = resnet50(weights=self.weights).to(self.device) 
        
        #set up the pre-processin pipeline 
        self.pre_process = self.weights.transforms()  
        
        #initialise the loss function
        self.loss = nn.CrossEntropyLoss() 
        
        #initialise the noise tensor
        self.noise = torch.zeros(3, 224, 224).to(self.device)
        
        
    def generate_adversarial_noise(self, img : torch.tensor, actual_class_index: int, target_class_index: int,
                                   lr=0.05, steps=500):
        
        """ 
        generate_adversarial_noise:
            Generates adversarial noise for an image.
            
        Args:
            img_path (str): path to the image file
            actual_index (int): the index of the actual class
            target_index (int): the index of the target class
            lr (float): the learning rate of the optimizer
            steps (int): the number of steps to take in the direction of the gradient 
            
        Returns:
            torch.Tensor: the adversarial noise tensor
            
        """
        
        #set the noise tensor to be a parameter that requires a gradient 
        noise = self.noise.detach().requires_grad_() 
        
        #set up the optimizer
        optimizer = torch.optim.Adam([noise], lr=lr) 
        
        #get the one hot encoded vector for the actual class 
        actual_class_index = F.one_hot(torch.tensor(actual_class_index), num_classes=1000).to(torch.float32).to(self.device)
        
        #get the one hot encoded vector the target class 
        adversarial_class = F.one_hot(torch.tensor(target_class_index), num_classes=1000).to(torch.float32).to(self.device)
        
        #iterate over the number of steps
        for iteration in range(steps): 
            
            #compute the adversarial image 
            adversarial_img = img + noise 
            
            #make a classification of the adversarial image 
            class_pred = self.model(adversarial_img.unsqueeze(0)).squeeze(0).softmax(0) 
            
            #compute the negative of the loss for the actual class by computing the negative we ensure the total loss will minimise the likelihood of this class 
            original_loss = - self.loss(class_pred, actual_class_index) 
            
            #compute the loss for the adversarial class this will maximise the likelihood of the adversarial class
            adversarial_loss = self.loss(class_pred, adversarial_class)
            
            #compute the total loss 
            loss = original_loss + adversarial_loss 
            
            #zero the gradients
            optimizer.zero_grad()
            #backpropagate the loss
            loss.backward() 
            #take a step in the direction of the gradient 
            optimizer.step() 
            
            predicted_class = self.weights.meta["categories"][class_pred.argmax().item()]
            
            #check if the adversarial class has been found
            if class_pred.argmax().item() == target_class_index:
                #if it has print the number of steps taken and break out of the loop
                print(f"Adversarial Noise Found in {iteration} steps")
                break
            
        return noise.detach(), predicted_class
    
    def produce_adversarial_image(self, image_path, target_class, save_path):
        """
        produce_adversarial_image: 
            produces an adversarial image and saves it to the outputs folder given an image and a target class. 
            
        Args:
            img_path (str): path to the image file
            actual_index (int): the index of the actual class
            target_index (int): the index of the target class
            
        Returns:
            None 
        
        """
        
        #load in the image and pre-process it 
        img = self.pre_process(Image.open(image_path)).to(self.device) 
        
        #get the logits associated with the image of the by making a classification with the model 
        logits = self.model(img.unsqueeze(0)).squeeze(0).softmax(0)
    
        #find the index of the 'actual' class    
        actual_class_index = logits.argmax().item()
        
        #get the classification index of the target class  
        target_class_index = self.weights.meta["categories"].index(target_class) 
        
        #generate the adversarial noise 
        noise, predicted_class = self.generate_adversarial_noise(img=img, actual_class_index=actual_class_index, target_class_index=target_class_index) 
        
        #take noise off the gpu and visualise it 
        noise = noise
        
        #add the noise to the image 
        adversarial_img = (img + noise).permute(1, 2, 0).cpu().detach().numpy()
        
        #normalise the image 
        adversarial_img = (adversarial_img - adversarial_img.min()) / (adversarial_img.max() - adversarial_img.min()) 
        
        
        #plot the adversarial image 
        plt.imshow(adversarial_img)
        plt.axis('off')
        plt.title(f'Predicted Class: {predicted_class}') 
        
        #save the image to the save path 
        plt.savefig(save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parse the input arguments 
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--target_class', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args() 
    
    #test the visualise image and noise function we now have an example adversarial image and noise which can be viewed in the output folder
    Adversary().produce_adversarial_image(image_path=args.image_path, save_path = args.save_path, target_class = args.target_class)  