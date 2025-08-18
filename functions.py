import cupy as cp
import matplotlib.pyplot as plt
wrap_kernel = cp.RawKernel(r'''
extern "C" __global__ void __global__ wrap(float2 *f, int n, int nz, int m)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
		return;
	if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
	{
		int tx0 = (tx - m + 2 * n) % (2 * n);
		int ty0 = (ty - m + 2 * n) % (2 * n);
		int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
		int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
		f[id1].x = f[id2].x;
		f[id1].y = f[id2].y;
	}
}
                           
''', 'wrap')

wrapadj_kernel = cp.RawKernel(r'''                
extern "C" __global__ void wrapadj(float2 *f, int n, int nz, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;
  if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
  {
    int tx0 = (tx - m + 2 * n) % (2 * n);
    int ty0 = (ty - m + 2 * n) % (2 * n);
    int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
    int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
    
    atomicAdd(&f[id2].x, f[id1].x);
    atomicAdd(&f[id2].y, f[id1].y);
  }
}

''', 'wrapadj')

gather_kernel = cp.RawKernel(r'''
extern "C" __global__ void gather(float2 *g, float2 *f, float *theta, int m,
                       float *mu, int n, int ntheta, int nz, bool direction)
{

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= ntheta || tz >= nz)
    return;
  float M_PI = 3.141592653589793238f;
  float2 g0, g0t;
  float w, coeff0;
  float w0, w1, x0, y0, coeff1;
  int ell0, ell1, g_ind, f_ind;

  g_ind = tx + ty * n + tz * n * ntheta;
  if (direction == 0) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x;
    g0.y = g[g_ind].y;
  }

  coeff0 = M_PI / mu[0];
  coeff1 = -M_PI * M_PI / mu[0];
  x0 = (tx - n / 2) / (float)n * __cosf(theta[ty]);
  y0 = -(tx - n / 2) / (float)n * __sinf(theta[ty]);
  if (x0 >= 0.5f)
    x0 = 0.5f - 1e-5;
  if (y0 >= 0.5f)
    y0 = 0.5f - 1e-5;
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;                             
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      w0 = ell0 / (float)(2 * n) - x0;
      w1 = ell1 / (float)(2 * n) - y0;
      w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1)); 
      f_ind = n + m + ell0 + (2 * n + 2 * m) * (n + m + ell1) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
      if (direction == 0) {
        g0.x += w * f[f_ind].x;
        g0.y += w * f[f_ind].y;
      } else {
        float *fx = &(f[f_ind].x);
        float *fy = &(f[f_ind].y);
        atomicAdd(fx, w * g0.x);
        atomicAdd(fy, w * g0.y);
      }
    }
  }
  if (direction == 0){
    g[g_ind].x = g0.x / n;
    g[g_ind].y = g0.y / n;
  }
}

''', 'gather')

def _init(nz, n):
    # usfft parameters
    eps = 1e-5  # accuracy of usfft
    mu = -cp.log(eps) / (2 * n * n)
    m = int(cp.ceil(2 * n * 1 / cp.pi * cp.sqrt(-mu *
            cp.log(eps) + (mu * n) * (mu * n) / 4)))
    # extra arrays
    # interpolation kernel
    t = cp.linspace(-1/2, 1/2, n, endpoint=False).astype('float32')
    [dx, dy] = cp.meshgrid(t, t)
    phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype('float32')) * (1-n % 4)

    # padded fft, reusable by chunks
    fde = cp.zeros([nz, 2*m+2*n, 2*m+2*n], dtype='complex64')
    # (+1,-1) arrays for fftshift
    c1dfftshift = (1-2*((cp.arange(1, n+1) % 2))).astype('int8')
    c2dtmp = 1-2*((cp.arange(1, 2*n+1) % 2)).astype('int8')
    c2dfftshift = cp.outer(c2dtmp, c2dtmp)
    return m, mu, phi, fde, c1dfftshift, c2dfftshift


def _wint(n, t):

    N = len(t)
    s = cp.linspace(1e-40, 1, n)
    # Inverse vandermonde matrix
    tmp1 = cp.arange(n)
    tmp2 = cp.arange(1, n+2)
    iv = cp.linalg.inv(cp.exp(cp.outer(tmp1, cp.log(s))))
    u = cp.diff(cp.exp(cp.outer(tmp2, cp.log(s)))*cp.tile(1.0 /
                tmp2[..., cp.newaxis], [1, n]))  # integration over short intervals
    W1 = cp.matmul(iv, u[1:n+1, :])  # x*pn(x) term
    W2 = cp.matmul(iv, u[0:n, :])  # const*pn(x) term

    # Compensate for overlapping short intervals
    tmp1 = cp.arange(1, n)
    tmp2 = (n-1)*cp.ones((N-2*(n-1)-1))
    tmp3 = cp.arange(n-1, 0, -1)
    p = 1/cp.concatenate((tmp1, tmp2, tmp3))
    w = cp.zeros(N)
    for j in range(N-n+1):
        # Change coordinates, and constant and linear parts
        W = ((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2

        for k in range(n-1):
            w[j:j+n] = w[j:j+n] + p[j+k]*W[:, k]

    wn = w
    wn[-40:] = (w[-40])/(N-40)*cp.arange(N-40, N)
    return wn

def calc_filter(n, filter):
    d = 0.5
    t = cp.arange(0, n/2+1)/n
    if filter == 'ramp':
        wfa = n*0.5*_wint(12, t)
    elif filter == 'shepp':
        wfa = n*0.5*_wint(12, t)*cp.sinc(t/(2*d))*(t/d <= 2)
    elif filter == 'cosine':
        wfa = n*0.5*_wint(12, t)*cp.cos(cp.pi*t/(2*d))*(t/d <= 1)
    elif filter == 'cosine2':
        wfa = n*0.5*_wint(12, t) * \
            (cp.cos(cp.pi*t/(2*d)))**2*(t/d <= 1)
    elif filter == 'hamming':
        wfa = n*0.5 * \
            _wint(12, t)*(.54 + .46 * cp.cos(cp.pi*t/d))*(t/d <= 1)
    elif filter == 'hann':
        wfa = n*0.5*_wint(12, t) * \
            (1+cp.cos(cp.pi*t/d)) / 2.0*(t/d <= 1)
    elif filter == 'parzen':
        wfa = n*0.5*_wint(12, t)*pow(1-t/d, 3)*(t/d <= 1)

    wfa = 2*n*wfa*(wfa >= 0)
    wfa[0] *= 2
    wfa = wfa.astype('float32')
    return wfa


def fbp_filter(data, filter_type='parzen'):
    """FBP filtering of projections"""

    n = data.shape[-1]
    ne = 4*n
    if filter_type!='none':
      tmp = cp.pad(data, ((0, 0), (0, 0), (ne//2-n//2, ne//2-n//2)), mode='edge')    
      wfilter = calc_filter(ne,filter_type)      
      wfilter = cp.concatenate((wfilter[:-1],wfilter[1:][::-1] ))              
      tmp = cp.fft.ifft(cp.fft.fft(tmp)*wfilter)
      data[:] = tmp[:, :, ne//2-n//2:ne//2+n//2]
    return data  # reuse input memory
    
def R(obj, theta, rotation_axis):
    """Radon transform for tomography projection
    Parameters
    ----------
    obj : ndarray
        Input 3D object, shape [nz,n,n]    
    theta : ndarray
        Projection angles, shape [ntheta]    
    rotation_axis : float
        Rotation axis 

    Returns
    -------
    sino : ndarray
        Output sinograms, shape [nz,ntheta,n]    
    """
    
    [nz, n, n] = obj.shape
    theta = cp.array(theta, dtype='float32')
    ntheta = len(theta)
    m, mu, phi, fde, c1dfftshift, c2dfftshift = _init(nz, n)

    sino = cp.zeros([nz, ntheta, n], dtype='complex64')

    # STEP0: multiplication by phi, padding
    fde = obj*phi
    fde = cp.pad(fde, ((0, 0), (n//2, n//2), (n//2, n//2)))

    # STEP1: fft 2d
    fde = cp.fft.fft2(fde*c2dfftshift)*c2dfftshift
    fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
    # STEP2: fft 2d
    wrap_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))
    mua = cp.array([mu], dtype='float32')
    gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32)), nz),
                  (32, 32, 1), (sino, fde, theta, m, mua, n, ntheta, nz, 0))

    # STEP3: ifft 1d
    sino = cp.fft.ifft(c1dfftshift*sino)*c1dfftshift

    # STEP4: Shift based on the rotation axis
    t = cp.fft.fftfreq(n).astype('float32')
    w = cp.exp(2*cp.pi*1j*t*(-rotation_axis + n/2))
    sino = cp.fft.ifft(w*cp.fft.fft(sino))
    # normalization for the unity test
    c = 4*n*cp.sqrt(n*ntheta)
    sino /= c
    return sino


def RT(sino, theta, rotation_axis,filter_type='none'):
    """Radon transform for tomography projection
    Parameters
    ----------
    obj : ndarray
        Input sinograms, shape [nz,ntheta,n]    
    theta : ndarray
        Projection angles, shape [ntheta]    
    rotation_axis : float
        Rotation axis 

    Returns
    -------
    obj : ndarray
        Output 3D object, shape [nz,n,n]
    """
    
    [nz, ntheta, n] = sino.shape
    theta = cp.array(theta, dtype='float32')

    m, mu, phi, fde, c1dfftshift, c2dfftshift = _init(nz, n)

    sino = fbp_filter(sino.copy(),filter_type)
    # STEP0: Shift based on the rotation axis
    t = cp.fft.fftfreq(n).astype('float32')
    w = cp.exp(-2*cp.pi*1j*t*(-rotation_axis + n/2))
    sino = cp.fft.ifft(w*cp.fft.fft(sino))

    # STEP1: fft 1d
    sino = cp.fft.fft(c1dfftshift*sino)*c1dfftshift

    # STEP2: interpolation (gathering) in the frequency domain
    # dont understand why RawKernel cant work with float, I have to send it as an array (TODO)
    mua = cp.array([mu], dtype='float32')
    gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32)), nz),
                  (32, 32, 1), (sino, fde, theta, m, mua, n, ntheta, nz, 1))
    wrapadj_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                    int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))

    # STEP3: ifft 2d
    fde = fde[:, m:-m, m:-m]
    fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift

    # STEP4: unpadding, multiplication by phi
    fde = fde[:, n//2:3*n//2, n//2:3*n//2]*phi
    c = n*cp.sqrt(n*ntheta)  # normalization for the unity test
    fde/=c
    
    return fde

def mshow_complex(a, show=True, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        im = axs[0].imshow(a.real, cmap="gray", **args)
        axs[0].set_title("real")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(a.imag, cmap="gray", **args)
        axs[1].set_title("imag")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()
  

def mshow(a, show=True, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()