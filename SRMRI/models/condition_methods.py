from abc import ABC,abstractmethod
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn import functional as F

__CONDITION_METHODS__ = {}

def register_condition_method(name):
    def wrapper(cls):
        if __CONDITION_METHODS__.get(name,None):
            raise NameError(f"Condition method {name} already exists")
        __CONDITION_METHODS__[name] = cls
        return cls
    return wrapper

def get_condition_method(name:str,operator, noise, **kwargs):
    if not __CONDITION_METHODS__.get(name,None):
        raise NameError(f"Condition method {name} does not exist")
    return __CONDITION_METHODS__[name](operator, noise, **kwargs)


class ConditionMethod(ABC):
    def __init__(self,operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

        
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            #Y-Ax
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            #||Y-Ax||^2
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_condition_method(name='dps')
class PosteriorSampling(ConditionMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_next=x_t -norm_grad * self.scale
        return x_next, norm

@register_condition_method(name='wavelet')  
class WaveletTransform:
    def __init__(self,operator,noiser,level,device):
       # Define wavelet transforms
        self.dwt = DWTForward(J=level, wave='bior4.4').to(device)
        self.idwt = DWTInverse(wave='bior4.4').to(device)
    
    def resize_image_torch(self,image, target_size1,target_size2):
        """Resize the image using PyTorch, expecting a tensor input."""
        # Assuming image is a torch tensor of shape (C, H, W)
        #image = image.unsqueeze(0)  # Add batch dimension
        resized_image = F.interpolate(image, size=(target_size1, target_size2), mode='bicubic', align_corners=False)
        return resized_image  # Remove batch dimension
    
    def conditioning(self,xt,measurement):
        #measurement= self.resize_image_torch(measurement, 720,512).float()  # Assuming y and xt are square and have a size of 720
        Yl_xt, Yh_xt = self.dwt(xt)
        Yl_y, Yh_y = self.dwt(measurement)
        fused_Yl = Yl_y  
        fused_Yh = Yh_xt  

        # Reconstruct the image from the fused coefficients
        fused_image = self.idwt((fused_Yl, fused_Yh))

        return fused_image


@register_condition_method(name='fourier')
class FourierTransform:
    def __init__(self, operator, noiser, device):
        self.device = device

    def conditioning(self, xt, measurement):
        # Image dimensions
        lr_shape = measurement.shape[-2:]
        hr_shape = xt.shape[-2:]
        
        # Scaling factor for normalization
        scale_factor = torch.sqrt(torch.tensor(lr_shape[0] * lr_shape[1], device=self.device, dtype=xt.dtype))

        # Compute the Fourier transform and shift the zero-frequency component to the center
        low_res_fft = torch.fft.fftshift(torch.fft.fftn(measurement, dim=(-2, -1))) / scale_factor
        high_res_fft = torch.fft.fftshift(torch.fft.fftn(xt, dim=(-2, -1))) / scale_factor
        
        center_lr = [dim // 2 for dim in lr_shape]
        center_hr = [dim // 2 for dim in hr_shape]
        
        # Calculate the range to replace in the high-res FFT with the low-res FFT
        start_lr = [0, 0]
        end_lr = lr_shape
        start_hr = [center_hr[0] - center_lr[0], center_hr[1] - center_lr[1]]
        end_hr = [start_hr[0] + lr_shape[0], start_hr[1] + lr_shape[1]]
        
        # Replace the low frequency components of high resolution with low resolution
        high_res_fft[..., start_hr[0]:end_hr[0], start_hr[1]:end_hr[1]] = low_res_fft[..., start_lr[0]:end_lr[0], start_lr[1]:end_lr[1]]

        # Shift back the zero-frequency component to the original place
        high_res_fft = torch.fft.ifftshift(high_res_fft)
        
        # Compute the inverse Fourier transform to get the consistent high resolution image
        consistent_high_res = torch.fft.ifftn(high_res_fft, dim=(-2, -1)) * scale_factor
        
        return torch.real(consistent_high_res)

@register_condition_method(name='mcg')
class ManifoldConstraintGradient(ConditionMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm

@register_condition_method(name='projection')
class Projection(ConditionMethod):
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t, None