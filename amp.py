import torch
import torch.nn as nn
import torch.nn.functional as F


class Autocast:

    def __init__(self, enabled=True, dtype=torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.target_dtype = dtype
        self._original_funcs = {}
        self._depth = 0

        # Define the functions we want to intercept and cast.
        # Format: (module, function_name_string)
        self.DOWNCAST_OPS = [
            (torch, 'matmul'),
            (torch, 'bmm'),
            (F, 'linear'),
        ]
        self.UPCAST_OPS = [
            (F, 'layer_norm'),
            (F, 'cross_entropy'),
        ]


    def _create_wrapper(self, original_func, target_dtype):
        """
        Creates a wrapper that casts inputs to the target dtype
        and calls the function with casted arguments
        """
        def wrapper(*args, **kwargs):
            """
            Wrapper that casts all inputs to the target dtype 
            and calls the function with casted arguments
            """

            # Cast all tensor arguments in args and kwargs
            ### YOUR CODE HERE
            new_args = tuple(
                arg.to(target_dtype) if torch.is_tensor(arg) and arg.is_floating_point() else arg
                for arg in args
            )

            new_kwargs = {
                key : (
                    val.to(target_dtype) if torch.is_tensor(val) and val.is_floating_point() else val
                )
                for key, val in kwargs.items()
            }

            # Call the original function with potentially casted inputs
            ### YOUR CODE HERE
            return original_func(*new_args, **new_kwargs)

        return wrapper

    def __enter__(self):
        """
        Wraps all the functions from DOWNCAST_OPS and UPCAST_OPS,
        Stores original functions
        And sets wrapped functions instead of original ones in the module
        """
        if not self.enabled:
            return self

        self._depth += 1
        if self._depth > 1:
            return self

        # Store original functions and apply patches
        for module, func_name in self.DOWNCAST_OPS + self.UPCAST_OPS:
            # Store original function
            ### YOUR CODE HERE
            original_func = getattr(module, func_name)
            self._original_funcs[(module, func_name)] = original_func

            # Create wrapped version of the function
            # Note that you need different target_dtype for DOWNCAST_OPS and UPCAST_OPS
            ### YOUR CODE HERE
            if (module, func_name) in self.DOWNCAST_OPS:
                target_dtype = self.target_dtype
            else:
                target_dtype = torch.float32

            wrapped_func = self._create_wrapper(original_func, target_dtype)


            # Set wrapped function as attribute of the module with the same name as original function
            ### YOUR CODE HERE
            setattr(module, func_name, wrapped_func)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores original function
        """
        if not self.enabled:
            return

        self._depth -= 1
        if self._depth > 0:
            return False

        # Restore original functions
        ### YOUR CODE HERE
        for (module, func_name), original_func in self._original_funcs.items():
            setattr(module, func_name, original_func)
        
        # Clear the stored functions for the next use
        ### YOUR CODE HERE
        self._original_funcs.clear()
        return False


class StaticGradScaler:
    def __init__(self, scale):
        """
        scale: loss scaling coef
        """
        self.scale_val = scale
    
    def scale(self, loss):
        """Scales the loss"""
        ### YOUR CODE HERE
        return loss * self.scale_val
    
    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        ### YOUR CODE HERE
        grads = []

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grads.append(param.grad)

        for grad in grads:
            if not torch.isfinite(grad).all():
                return

        for grad in grads:
            grad.div_(self.scale_val)

        optimizer.step()
    
    def update(self):
        """Updates scaling coef"""
        pass


class DynamicGradScaler:
    def __init__(self, scale, factor, patience, min_scale, max_scale):
        """
        scale: initial value of loss scaling coef
        factor: multiplier that used for scaling coef increase/decrease
        patience: how many iters there should be no nan/inf to increase scale
        min_scale: minimal allowed scaling coef value
        max_scale: maximal allowed scaling coef value
        """
        ### YOUR CODE HERE
        self.scale_value = scale
        self.factor = factor
        self.patience = patience
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.good_steps = 0
        self.inf = False

    def scale(self, loss):
        """Scales the loss"""
        ### YOUR CODE HERE
        return loss * self.scale_value

    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # If there is any nan/inf in the gradient decrease scaling coef using factor
        # Note that scaling coef should be greater than min_scale
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        ### YOUR CODE HERE
        grads = []
        
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grads.append(param.grad)

        self.inf = False
        for grad in grads:
            if not torch.isfinite(grad).all():
                self.inf = True
                break

        if self.inf:
            self.scale_value = max(self.min_scale, self.scale_value / self.factor)
            self.good_steps = 0

            return

        for grad in grads:
            grad.div_(self.scale_value)

        optimizer.step()

        self.good_steps += 1


    def update(self):
        """Updates scaling coef"""
        # If there was no any nan/inf in the gradient patience steps, increase scaling coef using factor
        # Note that scaling coef should be smaller than max_scale
        ### YOUR CODE HERE
        if self.inf:
            self.inf = False
            return

        if self.good_steps >= self.patience:
            self.scale_value = min(self.max_scale, self.scale_value * self.factor)
            self.good_steps = 0
