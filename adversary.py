import torch 
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights  
import torch.nn as nn
from torch.nn import functional as F 
import matplotlib.pyplot as plt


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
        
        
    def generate_adversarial_noise(self, img : torch.tensor, actual_index: int, target_index: int,
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
        actual_class = F.one_hot(torch.tensor(actual_index), num_classes=1000).to(torch.float32).to(self.device)
        
        #get the one hot encoded vector the target class 
        adversarial_class = F.one_hot(torch.tensor(target_index), num_classes=1000).to(torch.float32).to(self.device)
        
        #iterate over the number of steps
        for iteration in range(steps): 
            
            #compute the adversarial image 
            adversarial_img = img + noise 
            
            #make a classification of the adversarial image 
            class_pred = self.model(adversarial_img.unsqueeze(0)).squeeze(0).softmax(0) 
            
            #compute the negative of the loss for the actual class by computing the negative we ensure the total loss will minimise the likelihood of this class 
            original_loss = - self.loss(class_pred, actual_class) 
            
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
            
            #check if the adversarial class has been found
            if class_pred.argmax().item() == target_index:
                #if it has print the number of steps taken and break out of the loop
                print(f"Adversarial Noise Found in {iteration} steps")
                break
            
        return noise.detach()  
    
    
    def visualise_img_and_noise(self, img_path, actual_index, target_index):
        """
        
        visualise_img_and_noise: 
            Visualises the original image, the adversarial noise and the adversarial image. 
            
        Args:
            img_path (str): path to the image file
            actual_index (int): the index of the actual class
            target_index (int): the index of the target class
            
        Returns:
            None 
        
        """
        
        #load in the image and pre-process it 
        img = self.pre_process(Image.open(img_path)).to(self.device) 
        
        #generate the adversarial noise 
        noise = self.generate_adversarial_noise(img=img, actual_index=actual_index, target_index=target_index) 

        
        #plot the original image, the noise and the adversarial image        
        fig, ax = plt.subplots(1, 3) 
        
        ax[0].imshow(img.permute(1, 2, 0).cpu().detach().numpy())
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(noise.permute(1, 2, 0).cpu().detach().numpy())
        ax[1].set_title('Adversarial Noise')
        ax[1].axis('off')
        ax[2].imshow((img + noise).permute(1, 2, 0).cpu().detach().numpy())
        ax[2].set_title('Adversarial Image')
        ax[2].axis('off') 
    
        plt.savefig('outputs/adversarial_example.png')
        

#some example images have been downloaded from kaggle at: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000 this is one: renamed to giant_panda.jpg
test_img = 'inputs/giant_panda.jpg' 

if __name__ == "__main__":
    #test the visualise image and noise function we now have an example adversarial image and noise which can be viewed in the output folder
    Adversary().visualise_img_and_noise(img_path=test_img, actual_index=388, target_index=1)  