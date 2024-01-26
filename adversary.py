import torch 
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights  

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
        

#some example images have been downloaded from kaggle at: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000 this is one 
test_img = 'images/n01695060_1676.jpg' 

Adversary().classify(test_img)
