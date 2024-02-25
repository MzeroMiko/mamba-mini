# Modified by Mzero #20240123
# Copyright (C) 2023, Tri Dao, Albert Gu.
import math
from functools import partial
import torch
import torch.nn.functional as F
import pytest
from einops import rearrange, repeat

MODE = "v0"
# MODE = "fn"
# MODE = "fnDEBUG"

def selective_scan_easy(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
    """
    # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    us: B, G * D, L 
    dts: B, G * D, L
    As: G * D, N
    Bs: B, G, N, L
    Cs: B, G, N, L
    Ds: G * D
    delta_bias: G * D
    # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    """
    def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
        """
        partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
        => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
        => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
        => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
           y_i = C_i*h_i + D*u_i
        """
        """
        us, dts: (L, B, G, D) # L is chunk_size
        As: (G, D, N)
        Bs, Cs: (L, B, G, N)
        Ds: (G, D)
        hprefix: (B, G, D, N)
        """
        ts = dts.cumsum(dim=0)
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
        # scale = Ats[-1].detach()
        scale = 1
        rAts = Ats / scale
        duts = dts * us
        dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs)
        hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0) 
        hs = hs_tmp + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs) 
        return ys, hs
    

    dtype = torch.float32
    # dtype = torch.float16
    inp_dtype = us.dtype
    has_D = Ds is not None
    if chunksize < 1:
        chunksize = Bs.shape[-1]

    dts = dts.to(dtype)
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).to(dtype)
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    
    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
    As = As.view(G, -1, N).to(dtype)
    Bs = Bs.permute(3, 0, 1, 2).to(dtype)
    Cs = Cs.permute(3, 0, 1, 2).to(dtype)
    Ds = Ds.view(G, -1).to(dtype) if has_D else None
    D = As.shape[1]
    
    oys = []
    hprefix = us.new_zeros((B, G, D, N), dtype=dtype)
    for i in range(0, L, chunksize):
        ys, hs = selective_scan_chunk(
            us[i:i + chunksize], dts[i:i + chunksize], 
            As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix, 
        )
        oys.append(ys)
        hprefix = hs[-1]

    oys = torch.cat(oys, dim=0)
    if has_D:
        oys = oys + Ds * us
    oys = oys.permute(1, 2, 3, 0).view(B, -1, L)

    # return oys, hprefix.view(B, G * D, N)
    return oys.to(inp_dtype) if not return_last_state else (oys.to(inp_dtype), hprefix.view(B, G * D, N).float())


class SelectiveScanEasy(torch.autograd.Function):
    # for debug, we use it as an orinary object
    DEBUG = (MODE == "fnDEBUG")

    if DEBUG:
        print("DEBUG here...", flush=True)
        saved_tensors = []
        
        @classmethod
        def save_for_backward(ctx, *args):
            ctx.saved_tensors = args

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
        has_D = Ds is not None
        dtype = torch.float32
        
        dts = dts.to(dtype)
        if delta_bias is not None:
            dts = dts + delta_bias.view(1, -1, 1).to(dtype)
        if delta_softplus:
            dts = torch.nn.functional.softplus(dts)

        B_squeeze = (len(Bs.shape) == 3)
        C_squeeze = (len(Cs.shape) == 3)
        if B_squeeze:
            Bs = Bs.unsqueeze(1)
        if C_squeeze:
            Cs = Cs.unsqueeze(1)
        B, G, N, L = Bs.shape
        us = us.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
        dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
        As = As.view(G, -1, N).to(dtype)
        Bs = Bs.permute(3, 0, 1, 2).to(dtype)
        Cs = Cs.permute(3, 0, 1, 2).to(dtype)
        Ds = Ds.view(G, -1).to(dtype) if has_D else None
        D = As.shape[1]
        
        ctx.shape = (B, G, D, N, L)
        ctx.delta_softplus = delta_softplus
        ctx.return_last_state = return_last_state
        ctx.chunksize = chunksize
        ctx.BC_squeeze = (B_squeeze, C_squeeze)
        save_for_backward = [us, dts, As, Bs, Cs, Ds, delta_bias]

        chunks = list(range(0, L, chunksize))
        oys = []
        ohs = []        
        hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
        for i in chunks:
            ts = dts[i:i+chunksize].cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
            # scale = Ats[-1:].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts[i:i + chunksize] * us[i:i + chunksize]
            dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
            tmp_dtBus_div_rAts = (dtBus / rAts)
            tmp_dtBus_div_rAts_cumsum = tmp_dtBus_div_rAts.cumsum(dim=0) 
            hs = rAts * tmp_dtBus_div_rAts_cumsum + Ats * hprefix.unsqueeze(0)
            ys = torch.einsum("lbgn,lbgdn->lbgd", Cs[i:i + chunksize], hs) 
            oys.append(ys)
            ohs.append(hs)
            hprefix = hs[-1]

        oys = torch.cat(oys, dim=0)
        ohs = torch.cat(ohs, dim=0)
        if has_D:
            oys = oys + Ds * us

        save_for_backward.extend([ohs])

        ctx.save_for_backward(*save_for_backward)
        
        oys = oys.permute(1, 2, 3, 0).view(B, -1, L)

        if getattr(ctx, "DEBUG", False):
            print("DEBUG here ..............", flush=True)
            oys.backward = partial(ctx.backward, ctx)
        
        return oys, hprefix.view(B, G * D, N)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, doys: torch.Tensor, *args):
        DEBUG = getattr(ctx, "DEBUG", False)
        us, dts, As, Bs, Cs, Ds, delta_bias, ohs = ctx.saved_tensors
                
        B, G, D, N, L = ctx.shape
        chunksize = ctx.chunksize
        delta_softplus = ctx.delta_softplus
        doys = doys.view(B, G, D, L).permute(3, 0, 1, 2)

        def rev_comsum(x):
            cum_sum = torch.cumsum(x, dim=0)
            return (x - cum_sum + cum_sum[-1:None])
        
        if DEBUG:
            dtype = torch.float32
            us = us.requires_grad_()
            dts = dts.requires_grad_()
            As = As.requires_grad_()
            Bs = Bs.requires_grad_()
            Cs = Cs.requires_grad_()
            Ds = Ds.requires_grad_() if Ds is not None else None
            delta_bias = delta_bias.requires_grad_() if delta_bias is not None else None
            ohs = ohs.requires_grad_()

        # copy forward again
        if DEBUG:
            has_D = Ds is not None
            
            tmp_fwd_dtBus = []
            tmp_fwd_rAts = []
            tmp_fwd_Ats = []
            tmp_fwd_dtBus_div_rAts_cumsum = []
            tmp_fwd_dtBus_div_rAts = []

            chunks = list(range(0, L, chunksize))
            oys = []
            ohs = []        
            hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
            for i in chunks:
                ts = dts[i:i+chunksize].cumsum(dim=0)
                Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
                # scale = Ats[-1:].detach()
                scale = 1
                rAts = Ats / scale
                duts = dts[i:i + chunksize] * us[i:i + chunksize]
                dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
                tmp_dtBus_div_rAts = (dtBus / rAts)
                tmp_dtBus_div_rAts_cumsum = tmp_dtBus_div_rAts.cumsum(dim=0) 
                hs = rAts * tmp_dtBus_div_rAts_cumsum + Ats * hprefix.unsqueeze(0)
                ys = torch.einsum("lbgn,lbgdn->lbgd", Cs[i:i + chunksize], hs) 
                oys.append(ys)
                ohs.append(hs)
                hprefix = hs[-1]

                tmp_fwd_dtBus_div_rAts_cumsum.append(tmp_dtBus_div_rAts_cumsum)
                tmp_fwd_dtBus_div_rAts.append(tmp_dtBus_div_rAts)
                tmp_fwd_dtBus.append(dtBus)
                tmp_fwd_rAts.append(rAts)
                tmp_fwd_Ats.append(Ats)

            oys = torch.cat(oys, dim=0)
            ohs = torch.cat(ohs, dim=0)
            if has_D:
                oys = oys + Ds * us

        if DEBUG:
            _oys = oys.requires_grad_()

        dus = None
        dDs = None
        if Ds is not None:
            dDs = torch.einsum("lbgd,lbgd->gd", doys, us).view(-1)
            dus = torch.einsum("lbgd,gd->lbgd", doys, Ds)

        chunks = list(range(0, L, chunksize))
        dAs = us.new_zeros((G, D, N), dtype=torch.float)
        dus = us.new_zeros((L, B, G, D), dtype=torch.float) if dus is None else dus
        ddts = us.new_zeros((L, B, G, D), dtype=torch.float)
        dBs = us.new_zeros((L, B, G, N), dtype=torch.float)
        dCs = us.new_zeros((L, B, G, N), dtype=torch.float)
        dhprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
        for i in chunks[::-1]:
            # forward procedure ================
            tmp_dts = dts[i:i+chunksize]
            ts = tmp_dts.cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
            scale = Ats[-1].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts[i:i + chunksize] * us[i:i + chunksize]
            dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
            dtBus_div_rAts = (dtBus / rAts)
            hs_minus_prefix_div_rAts = dtBus_div_rAts.cumsum(dim=0)
            
            # hs_minus_prefix = rAts * hs_minus_prefix_div_rAts
            
            # below code is not ok due to precision limitation...
            # use saved hs instead
            if False:
                hprefix = (hsuffix - hs_minus_prefix[-1]) / scale
                hs = hs_minus_prefix + Ats * hprefix.unsqueeze(0)

            # backward procedure ================
            _hs = ohs[i:i+chunksize]
            _hprefix = ohs[i - 1] if i > 0 else None
            dCs[i:i + chunksize] = torch.einsum("lbgd,lbgdn->lbgn", doys[i:i + chunksize], _hs)
            dhs = doys[i:i + chunksize].unsqueeze(4) * Cs[i:i + chunksize].unsqueeze(3) # lbgd,lbgn->lbgdn
            dhs[-1] = dhs[-1] + dhprefix
            dhprefix = torch.einsum("lbgdn,lbgdn -> bgdn", dhs, Ats)
            dAts_hprefix = dhs * _hprefix.unsqueeze_(0) if i > 0 else None # lbgdn,bgdn->lbgdn
            drAts_hs_minus_prefix = dhs * hs_minus_prefix_div_rAts
            dhs_minus_prefix_div_rAts = dhs * rAts
            
            if DEBUG:
                print("1", (torch.autograd.grad(_oys, tmp_fwd_dtBus_div_rAts_cumsum[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0] - dhs_minus_prefix_div_rAts).abs().sum())

            d_dtBus_div_rAts = rev_comsum(dhs_minus_prefix_div_rAts)
            if DEBUG:
                d_dtBus_div_rAts_v1 = torch.autograd.grad(hs_minus_prefix_div_rAts, dtBus_div_rAts, dhs_minus_prefix_div_rAts, create_graph=True, allow_unused=True)[0]
                print("2.0", (torch.autograd.grad(_oys, tmp_fwd_dtBus_div_rAts[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0] - d_dtBus_div_rAts).abs().sum())
                print("2.1", (d_dtBus_div_rAts - d_dtBus_div_rAts_v1).abs().sum())
                d_dtBus_div_rAts = d_dtBus_div_rAts_v1
            
            ddtBus = d_dtBus_div_rAts / rAts
            dBs[i:i + chunksize] = torch.einsum("lbgdn,lbgd->lbgn", ddtBus, duts)
            dduts = torch.einsum("lbgdn,lbgn->lbgd", ddtBus, Bs[i:i + chunksize])
            dus[i:i + chunksize] = dus[i:i + chunksize] + dduts * dts[i:i + chunksize]
            if DEBUG:
                print("3", (torch.autograd.grad(_oys, tmp_fwd_dtBus[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0] - ddtBus).abs().sum())

            if DEBUG:
                tmp_a = torch.randn((L, B, G, D, N)).to(dtype).cuda().requires_grad_()
                tmp_b = torch.cumsum(tmp_a, dim=0)
                tmp_c = torch.randn((L, B, G, D, N)).to(dtype).cuda()
                print("ex.0", (torch.autograd.grad(tmp_b, tmp_a, tmp_c, create_graph=True, allow_unused=True)[0] - rev_comsum(tmp_c)).abs().sum())

            drAts_dtBus_div_rAts = d_dtBus_div_rAts * (-dtBus_div_rAts / rAts)
            if DEBUG:
                drAts_dtBus_div_rAts_v1 = d_dtBus_div_rAts * (dtBus / -(rAts * rAts)) # do not use this!!!
                drAts_dtBus_div_rAts_ref = torch.autograd.grad(dtBus_div_rAts, rAts, d_dtBus_div_rAts, create_graph=True, allow_unused=True)[0]
                print("4.0", (drAts_dtBus_div_rAts - drAts_dtBus_div_rAts_ref).abs().sum())
                print("4.0_v1", (drAts_dtBus_div_rAts - drAts_dtBus_div_rAts_v1).abs().sum())

            ddts[i:i + chunksize] = dduts * us[i:i + chunksize]
            dAts = drAts_dtBus_div_rAts / scale + drAts_hs_minus_prefix / scale + (dAts_hprefix if i > 0 else 0)

            if DEBUG:
                drAts_ref = torch.autograd.grad(_oys, tmp_fwd_rAts[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0]
                dAts_ref = torch.autograd.grad(_oys, tmp_fwd_Ats[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0]
                print("4.1", (drAts_ref - (drAts_dtBus_div_rAts + drAts_hs_minus_prefix)).abs().sum())
                print("4.2", ((drAts_ref - (drAts_dtBus_div_rAts + drAts_hs_minus_prefix)) / scale).abs().sum())
                print("4.3", (dAts_ref - dAts).abs().sum())
                
            dAts_noexp = dAts * Ats # d(e^x) = e^x * dx
            dAs = dAs + torch.einsum("lbgdn,lbgd->gdn", dAts_noexp, ts)
            d_ts = torch.einsum("lbgdn,gdn->lbgd", dAts_noexp, As)
            
            _part_ddts = rev_comsum(d_ts) # the precision is enough
            if DEBUG:
                _part_ddts_v1 = torch.autograd.grad(ts, tmp_dts, d_ts, create_graph=True, allow_unused=True)[0]
                print("5.0", (_part_ddts - _part_ddts_v1).abs().sum())

            ddts[i:i + chunksize] = ddts[i:i + chunksize] + _part_ddts
        
        if DEBUG:
            print("f", (torch.autograd.grad(_oys, dts, doys, create_graph=True, allow_unused=True)[0] - ddts).abs().sum(), flush=True)
        
        if delta_softplus:
            # softplus = log(1 + e^x); dsoftplus = e^x /(1+e^-x) = 1 - 1 / (1+e^x) = 1 - (e^-softplus)
            ddts = ddts - ddts * (-dts).exp()

        ddelta_bias = None 
        if delta_bias is not None:
            ddelta_bias = ddts.sum([0, 1]).view(-1)
        
        if DEBUG:
            print("f", (torch.autograd.grad(_oys, us, doys, create_graph=True, allow_unused=True)[0] - dus).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, Bs, doys, create_graph=True, allow_unused=True)[0] - dBs).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, Cs, doys, create_graph=True, allow_unused=True)[0] - dCs).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, Ds, doys, create_graph=True, allow_unused=True)[0].view(-1) - dDs).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, As, doys, create_graph=True, allow_unused=True)[0] - dAs).abs().sum(), flush=True)
            # print("f", (torch.autograd.grad(_oys, delta_bias, doys, create_graph=True, allow_unused=True)[0] - ddelta_bias).abs().sum(), flush=True)
            
        dus = dus.permute(1, 2, 3, 0).view(B, -1, L)
        ddts = ddts.permute(1, 2, 3, 0).view(B, -1, L)
        dAs = dAs.view(-1, N)
        dBs = dBs.permute(1, 2, 3, 0)
        dCs = dCs.permute(1, 2, 3, 0)
        if ctx.BC_squeeze[0]:
            dBs = dBs.flatten(1, 2)
        if ctx.BC_squeeze[1]:
            dCs = dCs.flatten(1, 2)

        return dus, ddts, dAs, dBs, dCs, dDs, ddelta_bias, None, None, None


# api to fit original mamba_ssm
def build_api_selective_scan(chunksize=64):
    mode = MODE

    def selective_scan_fn(u, delta, A, B, C, D, z=None,
        delta_bias=None, delta_softplus=None,
        return_last_state=False):
        assert z is None
        if mode in ["fnDEBUG"]:
            outs = SelectiveScanEasy.forward(SelectiveScanEasy, u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state, chunksize)
            return (outs[0].to(u.dtype), *outs[1:]) if return_last_state else outs[0].to(u.dtype)
        elif mode in ["fn"]:
            outs = SelectiveScanEasy.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state, 1)
            return (outs[0].to(u.dtype), *outs[1:]) if return_last_state else outs[0].to(u.dtype)
        elif mode in ["v0"]:
            return selective_scan_easy(u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state, chunksize)
        else:
            raise NotImplementedError
    return selective_scan_fn


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# @pytest.mark.parametrize('wtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('delta_softplus', [False, True])
@pytest.mark.parametrize('has_z', [False])
@pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("chunksize", [64])
# @pytest.mark.parametrize("chunksize", [32])
def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seqlen, itype, wtype, chunksize):
    selective_scan_fn = build_api_selective_scan(chunksize=chunksize)

    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 24
    dstate = 8
    is_complex = wtype == torch.complex64
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                    requires_grad=True)
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_z:
        z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    else:
        z = None
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out, *rest = selective_scan_fn(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
    if return_last_state:
        state = rest[0]
    out_ref, *rest = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
    if return_last_state:
        state_ref = rest[0]
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        print(f'State max diff: {(state - state_ref).abs().max().item()}')
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_z:
        print(f'dz max diff: {(z.grad - z_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                          atol=atolw if not is_variable_B else atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                          atol=atolw if not is_variable_C else atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)


# pytest test_selective_scan.py

if __name__ == "__main__":
    test_selective_scan(True, True, 2, True, False, True, True, True, 512, torch.float32, torch.float32, 64)
