import torch
from model_core.ops import OPS_CONFIG
from model_core.factors import FeatureEngineer

class StackVM:
    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor):
        # feat_tensor: [N, F, T]
        N = feat_tensor.shape[0]
        T = feat_tensor.shape[2]
        ref = (N, T)

        def _fail(reason, **kw):
            info = {"reason": reason, **kw}
            return None, info

        def _coerce_shape(x, token, op_name):
            # 目标：尽量变成 [N, T]，否则判定 shape_error
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"op returned non-tensor: {type(x)}")

            if x.ndim == 0:
                x = x.view(1, 1).expand(N, T)
            elif x.ndim == 1:
                if x.shape[0] == N:
                    x = x.unsqueeze(1).expand(N, T)
                elif x.shape[0] == T:
                    x = x.unsqueeze(0).expand(N, T)
                else:
                    raise RuntimeError(f"bad 1d shape={tuple(x.shape)}")
            elif x.ndim == 2:
                if x.shape == ref:
                    pass
                elif x.shape[0] == N and x.shape[1] == 1:
                    x = x.expand(N, T)
                elif x.shape[0] == 1 and x.shape[1] == T:
                    x = x.expand(N, T)
                else:
                    # 允许可广播但不明确的情况就直接判错，避免静默 bug
                    raise RuntimeError(f"bad 2d shape={tuple(x.shape)}")
            else:
                raise RuntimeError(f"bad ndim={x.ndim}, shape={tuple(x.shape)}")

            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            return x

        stack = []
        try:
            for j, token in enumerate(formula_tokens):
                token = int(token)

                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])  # [N, T]
                    continue
                if token not in self.op_map:
                    return _fail("unknown_token", step=j, token=token, stack_depth=len(stack))
                arity = self.arity_map[token]
                if len(stack) < arity:
                    return _fail("stack_underflow", step=j, token=token, arity=arity, stack_depth=len(stack))
                args = [stack.pop() for _ in range(arity)][::-1]
                op_name = self.op_name_map.get(token, f"op_{token}")
                try:
                    out = self.op_map[token](*args)
                    out = _coerce_shape(out, token, op_name)
                except Exception as e:
                    return _fail("op_exception", step=j, token=token, op=op_name, err=repr(e), stack_depth=len(stack))
                stack.append(out)
            if len(stack) != 1:
                return _fail("stack_not_one", final_stack=len(stack))
            return stack[0], {"reason": "ok"}

        except Exception as e:
            return _fail("vm_exception", err=repr(e))