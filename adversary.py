import torch 
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights  
from torch.utils.tensorboard import SummaryWriter 
import torch.nn as nn
from torch.nn import functional as F 


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
        
        #initialise the logger the runs folder is in the gitignore to save space 
        self.logger = SummaryWriter('runs') 
        
        #initialise the noise tensor
        self.noise = torch.zeros(3, 224, 224).to(self.device)
        
        
          
    def classify(self, img_path: str):
        """ 
        classify: 
            Classifies an image using a pretrained ResNet50 model. 
            Prints the predicted class and the likelihood of the prediction. 
        
        Args:
            img_path (str): path to image file
            
        Returns:
            None    
        
        """
        #load in the image 
        img = Image.open(img_path)
        
        #preprocess the image 
        img = self.pre_process(img).to(self.device)
        
        #make a classification
        classification = self.model(img.unsqueeze(0)).squeeze(0).softmax(0)

        #compute the predicted class 
        predicted_class = classification.argmax().item()

        #get the likelihood of the prediction
        likelihood = classification[predicted_class].item()
        
        #get the class name 
        class_name = self.weights.meta["categories"][predicted_class]

        #print the results 
        print(f"Predicted class: {class_name} (With Confidence of {100 * likelihood})")  
        
        
        
    def generate_adversarial_noise(self, img_path : str, actual_index: int, target_index: int,
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
        
        #preprocess the image 
        img = self.pre_process(Image.open(img_path)).to(self.device) 
        
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
            
            #log the loss and the predicted class 
            self.logger.add_scalar('Loss', loss.item(), iteration) 
            self.logger.add_text('Predicted Class', self.weights.meta["categories"][class_pred.argmax().item()], iteration) 
            
            #check if the adversarial class has been found
            if class_pred.argmax().item() == target_index:
                #if it has print the number of steps taken and break out of the loop
                print(f"Adversarial Noise Found in {iteration} steps")
                break
            
        return noise.detach() 

#some example images have been downloaded from kaggle at: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000 this is one: renamed to giant_panda.jpg
test_img = 'images/giant_panda.jpg' 

if __name__ == "__main__":
    #test the generate_adversarial_noise function here the target class is a goldfish 
    Adversary().generate_adversarial_noise(test_img, 388, 1) 