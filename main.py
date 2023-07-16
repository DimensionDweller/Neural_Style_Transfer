import torch
from models.vgg import VGGNet, load_vgg
from utils.image_loader import load_image
from utils.image_transformer import im_convert, save_image
from losses.style_content_loss import get_style_loss, get_content_loss
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)

# Load VGG19 with pretrained ImageNet weights
vgg = load_vgg(device)
vggnet = VGGNet().to(device).eval()

# Load images
content = load_image('IMG_8844.JPG', max_size=400, device=device)
style = load_image('bob_ross_v2.jpeg', shape=[content.size(2), content.size(3)], device=device)

# Initialize a target image with the content image
target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999])

for step in range(2000):
    target_features = vggnet(target)
    content_features = vggnet(content)
    style_features = vggnet(style)
    style_loss = 0
    content_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss += get_content_loss(f1, f2)
        style_loss += get_style_loss(f1, f3)
    loss = content_loss + 100 * style_loss 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step+1) % 10 == 0:
        print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
               .format(step+1, 2000, content_loss.item(), style_loss.item()))

# Save the generated image
save_image(target, 'output.png')

plt.imshow(im_convert(target))
plt.axis('off')
