import sys
import numpy as np
import cdc_loader

def reconstitute(filename, fieldnum):
	if fieldnum=='x' or fieldnum=='y':
		app = cdc_loader.app(filename)
		nb = app.config['mesh']['num_blocks']
		bs = app.config['mesh']['block_size']
		x, y   = np.zeros([bs * nb+1, bs * nb+1]), np.zeros([bs * nb+1, bs * nb+1])
		for (i, j), data in getattr(app.state, 'sigma').items():
			x[i*bs:i*bs+bs+1, j*bs:j*bs+bs+1], y[i*bs:i*bs+bs+1, j*bs:j*bs+bs+1] = app.mesh[(i,j)].vertices
		if fieldnum=='x':
			return x
		if fieldnum=='y':
			return y
	if fieldnum==0:
		field='sigma'
	if fieldnum==1:
		field='velocity_x'
	if fieldnum==2:
		field='velocity_y'
	if fieldnum==3:
		field='pressure'
	if fieldnum==4:
		field='specific_internal_energy'
	if fieldnum==5:
		field='mach_number'
	app = cdc_loader.app(filename)
	nb = app.config['mesh']['num_blocks']
	bs = app.config['mesh']['block_size']
	result = np.zeros([bs * nb, bs * nb])
	x, y   = result*0, result*0
	for (i, j), data in getattr(app.state, field).items():
		result[i*bs:i*bs+bs, j*bs:j*bs+bs] = data
	return result

nstr       = str(np.char.zfill(str(Nchkpts[0]),4))
d          = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor')
DR         = d.config['mesh']['domain_radius']
N          = int(d.config['mesh']['num_blocks'] * d.config['mesh']['block_size'])
dx         = 2*DR/N
dy         = dx*1
rsof       = d.config['physics']['softening_length']
cfl        = d.config['physics']['cfl']
gamma      = d.config['hydro']['euler']['gamma_law_index']
coolcoef   = d.config['hydro']['euler']['cooling_coefficient']
buffrate   = d.config['physics']['buffer_rate']
buffscale  = d.config['physics']['buffer_scale']
sinkrad    = d.config['physics']['sink_radius']
sinkrate   = d.config['physics']['sink_rate']
lambdamult = d.config['physics']['lambda_multiplier']
m1,m2      = d.orbital_state[0][0], d.orbital_state[1][0]

visctype  = list(d.config['physics']['viscosity'].keys())[0]
if visctype=='constant_nu':
	nu_or_alpha = d.config['physics']['viscosity']['constant_nu']
elif visctype=='alpha':
	nu_or_alpha = d.config['physics']['viscosity']['alpha']
else:
	sys.exit("Viscosity type not recognized")

x         = np.arange((N))*dx - 2*DR/2. + dx/2.
xx,yy     = np.zeros((N,N)),np.zeros((N,N))
for i in range(N):
        xx[:,i] = x*1
        yy[i,:] = x*1
rr        = np.sqrt(xx**2+yy**2)
buffer_mask = np.ones((N,N))
for i in range(N):
	for j in range(N):
		if rr[i,j]>DR-buffscale:
			buffer_mask[i,j] = 0.0

def dery(v,dy):
	dv = v*0
	dv[:,1:-1] = v[:,2:] - v[:,:-2]
	dv[:,0]    = v[:,1 ] - v[:, -1]
	dv[:,  -1] = v[:,0 ] - v[:, -2]
	return dv / (2.*dy)

def derx(v,dx):
	dv = v*0
	dv[1:-1,:] = v[2:,:] - v[:-2,:]
	dv[0,   :] = v[1 ,:] - v[ -1,:]
	dv[-1,  :] = v[0 ,:] - v[ -2,:]
	return dv / (2.*dx)

def dert(v1,v2,t1,t2):
	return (v2-v1)/(t2-t1)

def momentum(rho,v):
	return rho*v

def energy(rho,vx,vy,eps):
	return rho*eps + 0.5*rho*(vx**2+vy**2)

def compute_dert( rho_nm1, rho_np1,\
		   vx_nm1,  vx_np1,\
		   vy_nm1,  vy_np1,\
		  eps_nm1, eps_np1,\
		    t_nm1,   t_np1):
	dt_rho_n  = dert(          rho_nm1         ,          rho_np1         , t_nm1, t_np1 )
	dt_momx_n = dert( momentum(rho_nm1, vx_nm1), momentum(rho_np1, vx_np1), t_nm1, t_np1 )
	dt_momy_n = dert( momentum(rho_nm1, vy_nm1), momentum(rho_np1, vy_np1), t_nm1, t_np1 )
	dt_en_n   = dert( energy(rho_nm1, vx_nm1, vy_nm1, eps_nm1), \
        	          energy(rho_np1, vx_np1, vy_np1, eps_np1), t_nm1, t_np1 )
	return dt_rho_n, dt_momx_n, dt_momy_n, dt_en_n

def inviscid_fluxes_for_derx( rho, vx, vy, pres, eps ):
	rhoflux  = rho*vx
	momxflux = rho*vx*vx + pres
	momyflux = rho*vy*vx
	enflux   = (rho*eps + 0.5*rho*(vx**2 + vy**2) + pres)*vx
	return rhoflux, momxflux, momyflux, enflux

def inviscid_fluxes_for_dery( rho, vx, vy, pres, eps ):
	rhoflux  = rho*vy
	momxflux = rho*vx*vy
	momyflux = rho*vy*vy + pres
	enflux   = (rho*eps + 0.5*rho*(vx**2 + vy**2) + pres)*vy
	return rhoflux, momxflux, momyflux, enflux

def viscid_fluxes_for_derx( rho, vx, vy, pres, eps, nu ):
	lam      = lambdamult * nu
	dxvx     = derx(vx,dx)
	dyvy     = dery(vy,dy)
	dxvy     = derx(vy,dx)
	dyvx     = dery(vx,dy)
	div      = dxvx + dyvy
	tau_xx   = rho * nu * ( 2*dxvx - (2./3)*div ) + rho * lam * div
	tau_yy   = rho * nu * ( 2*dyvy - (2./3)*div ) + rho * lam * div
	tau_xy   = rho * nu * ( dyvx + dxvy )
	tau_yx   = tau_xy*1
	momxflux = tau_xx
	momyflux = tau_xy
	enflux   = vx*tau_xx + vy*tau_xy
	return momxflux, momyflux, enflux

def viscid_fluxes_for_dery( rho, vx, vy, pres, eps, nu ):
	lam      = lambdamult * nu
	dxvx     = derx(vx,dx)
	dyvy     = dery(vy,dy)
	dxvy     = derx(vy,dx)
	dyvx     = dery(vx,dy)
	div      = dxvx + dyvy
	tau_xx   = rho * nu * ( 2*dxvx - (2./3)*div ) + rho * lam * div
	tau_yy   = rho * nu * ( 2*dyvy - (2./3)*div ) + rho * lam * div
	tau_xy   = rho * nu * ( dyvx + dxvy )
	tau_yx   = tau_xy*1
	momxflux = tau_yx
	momyflux = tau_yy
	enflux   = vx*tau_yx + vy*tau_yy
	return momxflux, momyflux, enflux

def kinematic_viscosity( rho, pres, nu_or_alpha, x, y, x1, y1, x2, y2, m1, m2):
	if visctype=='constant_nu':
		return nu_or_alpha
	elif visctype=='alpha':
		alpha = nu_or_alpha
		cs2   = gamma*pres/rho
		r1    = np.sqrt((x-x1)**2 + (y-y1)**2)
		r2    = np.sqrt((x-x2)**2 + (y-y2)**2)
		twof  = 1./np.sqrt(m1/r1**3 + m2/r2**3)
		return alpha*cs2*twof/np.sqrt(gamma)
	else:
		sys.exit("Viscosity type not recognized")

def compute_all_fluxes_der( rho, vx, vy, pres, eps, nu_or_alpha, x, y, x1, y1, x2, y2, m1, m2 ):
	nu = kinematic_viscosity(rho,pres,nu_or_alpha,x,y,x1,y1,x2,y2,m1,m2)
	rhoflux_invisc_x, momxflux_invisc_x, momyflux_invisc_x, enflux_invisc_x = inviscid_fluxes_for_derx( rho, vx, vy, pres, eps )
	rhoflux_invisc_y, momxflux_invisc_y, momyflux_invisc_y, enflux_invisc_y = inviscid_fluxes_for_dery( rho, vx, vy, pres, eps )
	momxflux_visc_x, momyflux_visc_x, enflux_visc_x = viscid_fluxes_for_derx( rho, vx, vy, pres, eps, nu )
	momxflux_visc_y, momyflux_visc_y, enflux_visc_y = viscid_fluxes_for_dery( rho, vx, vy, pres, eps, nu )
	d_rhoflux  = derx( rhoflux_invisc_x,dx) + dery( rhoflux_invisc_y,dy)
	d_momxflux = derx(momxflux_invisc_x,dx) + dery(momxflux_invisc_y,dy) - derx(momxflux_visc_x,dx) - dery(momxflux_visc_y,dy)
	d_momyflux = derx(momyflux_invisc_x,dx) + dery(momyflux_invisc_y,dy) - derx(momyflux_visc_x,dx) - dery(momyflux_visc_y,dy)
	d_enflux   = derx(  enflux_invisc_x,dx) + dery(  enflux_invisc_y,dy) - derx(  enflux_visc_x,dx) - dery(  enflux_visc_y,dy)
	return d_rhoflux, d_momxflux, d_momyflux, d_enflux


def sources_gravity( rho, vx, vy, x, y, x1, y1, x2, y2 ):
	r1   = np.sqrt((x-x1)**2 + (y-y1)**2)
	r2   = np.sqrt((x-x2)**2 + (y-y2)**2)
	fpre1= -rho * m1/(r1**2 + rsof**2)**(3./2)
	fpre2= -rho * m2/(r2**2 + rsof**2)**(3./2)
	f1x  = fpre1 * (x-x1)
	f1y  = fpre1 * (y-y1)
	f2x  = fpre2 * (x-x2)
	f2y  = fpre2 * (y-y2)
	momxsrc = f1x + f2x
	momysrc = f1y + f2y
	ensrc   = vx*f1x + vy*f1y + vx*f2x + vy*f2y
	return momxsrc, momysrc, ensrc

def sources_buffer( rho, vx, vy, pres, eps, rho0, vx0, vy0, pres0, eps0, x, y ):
	r = np.sqrt(x**2 + y**2)
	massbuff = (rho - rho0)
	momxbuff = (rho*vx - rho0*vx0)
	momybuff = (rho*vy - rho0*vy0)
	v2       = vx**2  + vy**2
	v20      = vx0**2 + vy0**2
	en       = rho *eps  + 0.5*rho *v2
	en0      = rho0*eps0 + 0.5*rho0*v20
	enbuff   = (en - en0)
	rbuff    = (r-DR)/buffscale
	omeg_out = np.sqrt(1./DR**3)
	buffwind = 0.5 * buffrate * (1.0 + np.tanh(rbuff)) * omeg_out
	massbuff*= -buffwind
	momxbuff*= -buffwind
	momybuff*= -buffwind
	enbuff  *= -buffwind
	return massbuff, momxbuff, momybuff, enbuff

def sources_cooling( rho, eps ):
	return -coolcoef * eps**4 / rho

def sources_sinks( rho, vx, vy, pres, eps, x, y, xbh, ybh ):
	momx = momentum(rho,vx)
	momy = momentum(rho,vy)
	en   = energy(rho,vx,vy,eps)
	rbh2 = (x-xbh)**2 + (y-ybh)**2
	s2   = sinkrad**2
	sinkwind = np.exp(-(rbh2/s2)**3) * sinkrate
	for i in range(N):
		for j in range(N):
			if rbh2[i,j] >= s2 * 9.0:
				sinkwind[i,j] = 0.0
	return -sinkwind * rho, -sinkwind * momx, -sinkwind * momy, -sinkwind * en


n       = Nchkpts[0]
nstr    = str(np.char.zfill(str(n),4))
rho_nm1 = reconstitute(fn+'chkpt.'+nstr+'.cbor',0)
vx_nm1  = reconstitute(fn+'chkpt.'+nstr+'.cbor',1)
vy_nm1  = reconstitute(fn+'chkpt.'+nstr+'.cbor',2)
pres_nm1= reconstitute(fn+'chkpt.'+nstr+'.cbor',3)
eps_nm1 = reconstitute(fn+'chkpt.'+nstr+'.cbor',4)
t_nm1   = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor').state.time

rho0    =  rho_nm1*1 #Keep for buffer source terms
vx0     =   vx_nm1*1
vy0     =   vy_nm1*1
pres0   = pres_nm1*1
eps0    =  eps_nm1*1

n       = Nchkpts[1]
nstr    = str(np.char.zfill(str(n),4))
d_n     = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor')
rho_n   = reconstitute(fn+'chkpt.'+nstr+'.cbor',0)
vx_n    = reconstitute(fn+'chkpt.'+nstr+'.cbor',1)
vy_n    = reconstitute(fn+'chkpt.'+nstr+'.cbor',2)
pres_n  = reconstitute(fn+'chkpt.'+nstr+'.cbor',3)
eps_n   = reconstitute(fn+'chkpt.'+nstr+'.cbor',4)
t_n     = d_n.state.time
x1_n,y1_n = d_n.orbital_state[0][1],d_n.orbital_state[0][2]
x2_n,y2_n = d_n.orbital_state[1][1],d_n.orbital_state[1][2]

t = []
t.append(t_n)

n       = Nchkpts[2]
nstr    = str(np.char.zfill(str(n),4))
d_np1   = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor')
rho_np1 = reconstitute(fn+'chkpt.'+nstr+'.cbor',0)
vx_np1  = reconstitute(fn+'chkpt.'+nstr+'.cbor',1)
vy_np1  = reconstitute(fn+'chkpt.'+nstr+'.cbor',2)
pres_np1= reconstitute(fn+'chkpt.'+nstr+'.cbor',3)
eps_np1 = reconstitute(fn+'chkpt.'+nstr+'.cbor',4)
t_np1   = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor').state.time
x1_np1,y1_np1 = d_np1.orbital_state[0][1],d_np1.orbital_state[0][2]
x2_np1,y2_np1 = d_np1.orbital_state[1][1],d_np1.orbital_state[1][2]

dt_rho_n, dt_momx_n, dt_momy_n, dt_en_n = compute_dert( rho_nm1, rho_np1,\
							 vx_nm1,  vx_np1,\
							 vy_nm1,  vy_np1,\
							eps_nm1, eps_np1,\
							  t_nm1,   t_np1)

d_rhoflux_n, d_momxflux_n, d_momyflux_n, d_enflux_n = compute_all_fluxes_der( rho_n, vx_n, vy_n, pres_n, eps_n, nu_or_alpha, xx, yy, x1_n, y1_n, x2_n, y2_n, m1, m2 )

momxsrc_grav, momysrc_grav, ensrc_grav = sources_gravity( rho_n, vx_n, vy_n, xx, yy, x1_n, y1_n, x2_n, y2_n )

rhosrc_buff, momxsrc_buff, momysrc_buff, ensrc_buff = sources_buffer( rho_n, vx_n, vy_n, pres_n, eps_n, rho0, vx0, vy0, pres0, eps0, xx, yy )

ensrc_cool = sources_cooling( rho_n, eps_n )

rhosrc_sink1, momxsrc_sink1, momysrc_sink1, ensrc_sink1 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x1_n, y1_n )
rhosrc_sink2, momxsrc_sink2, momysrc_sink2, ensrc_sink2 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x2_n, y2_n )

rhores   ,momxres   ,momyres   ,enres    = [],[],[],[]
rhores_L2,momxres_L2,momyres_L2,enres_L2 = [],[],[],[]

rhores. append( dt_rho_n  + d_rhoflux_n  -  rhosrc_buff -  rhosrc_sink1 - rhosrc_sink2                                )
momxres.append( dt_momx_n + d_momxflux_n - momxsrc_buff - momxsrc_grav  - momxsrc_sink1 - momxsrc_sink2               )
momyres.append( dt_momy_n + d_momyflux_n - momysrc_buff - momysrc_grav  - momysrc_sink1 - momysrc_sink2               )
enres.  append( dt_en_n   + d_enflux_n   -   ensrc_buff -   ensrc_grav  - ensrc_cool    -   ensrc_sink1 - ensrc_sink2 )

l,r=0,N

rhores_L2. append( np.sqrt(np.average(( rhores[-1][l:r,l:r]*buffer_mask)**2)) )
momxres_L2.append( np.sqrt(np.average((momxres[-1][l:r,l:r]*buffer_mask)**2)) )
momyres_L2.append( np.sqrt(np.average((momyres[-1][l:r,l:r]*buffer_mask)**2)) )
enres_L2.  append( np.sqrt(np.average((  enres[-1][l:r,l:r]*buffer_mask)**2)) )

for i in range(3,len(Nchkpts)-1):
	print("Analyzing checkpoint",i,"of",len(Nchkpts)-2)

	rho_nm1 =  rho_n*1
	vx_nm1  =   vx_n*1
	vy_nm1  =   vy_n*1
	pres_nm1= pres_n*1
	eps_nm1 =  eps_n*1
	t_nm1   =    t_n*1

	rho_n   =  rho_np1*1
	vx_n    =   vx_np1*1
	vy_n    =   vy_np1*1
	pres_n  = pres_np1*1
	eps_n   =  eps_np1*1
	t_n     =    t_np1*1
	t.append(t_n)
	x1_n,y1_n = x1_np1*1,y1_np1*1
	x2_n,y2_n = x2_np1*1,y2_np1*1

	n       = Nchkpts[i]
	nstr    = str(np.char.zfill(str(n),4))
	d_np1   = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor')
	rho_np1 = reconstitute(fn+'chkpt.'+nstr+'.cbor',0)
	vx_np1  = reconstitute(fn+'chkpt.'+nstr+'.cbor',1)
	vy_np1  = reconstitute(fn+'chkpt.'+nstr+'.cbor',2)
	pres_np1= reconstitute(fn+'chkpt.'+nstr+'.cbor',3)
	eps_np1 = reconstitute(fn+'chkpt.'+nstr+'.cbor',4)
	t_np1   = cdc_loader.app(fn+'chkpt.'+nstr+'.cbor').state.time
	x1_np1,y1_np1 = d_np1.orbital_state[0][1],d_np1.orbital_state[0][2]
	x2_np1,y2_np1 = d_np1.orbital_state[1][1],d_np1.orbital_state[1][2]

	dt_rho_n, dt_momx_n, dt_momy_n, dt_en_n = compute_dert( rho_nm1, rho_np1,\
								 vx_nm1,  vx_np1,\
								 vy_nm1,  vy_np1,\
								eps_nm1, eps_np1,\
								  t_nm1,   t_np1)

	d_rhoflux_n, d_momxflux_n, d_momyflux_n, d_enflux_n = compute_all_fluxes_der( rho_n, vx_n, vy_n, pres_n, eps_n, nu_or_alpha, xx, yy, x1_n, y1_n, x2_n, y2_n, m1, m2 )

	momxsrc_grav, momysrc_grav, ensrc_grav = sources_gravity( rho_n, vx_n, vy_n, xx, yy, x1_n, y1_n, x2_n, y2_n )

	rhosrc_buff, momxsrc_buff, momysrc_buff, ensrc_buff = sources_buffer( rho_n, vx_n, vy_n, pres_n, eps_n, rho0, vx0, vy0, pres0, eps0, xx, yy )

	ensrc_cool = sources_cooling( rho_n, eps_n )

	rhosrc_sink1, momxsrc_sink1, momysrc_sink1, ensrc_sink1 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x1_n, y1_n )
	rhosrc_sink2, momxsrc_sink2, momysrc_sink2, ensrc_sink2 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x2_n, y2_n )

	rhores, momxres, momyres, enres = [],[],[],[] #kill old lists, otherwise memory usage grows

	rhores. append( dt_rho_n  + d_rhoflux_n  -  rhosrc_buff -  rhosrc_sink1 - rhosrc_sink2                                )
	momxres.append( dt_momx_n + d_momxflux_n - momxsrc_buff - momxsrc_grav  - momxsrc_sink1 - momxsrc_sink2               )
	momyres.append( dt_momy_n + d_momyflux_n - momysrc_buff - momysrc_grav  - momysrc_sink1 - momysrc_sink2               )
	enres.  append( dt_en_n   + d_enflux_n   -   ensrc_buff -   ensrc_grav  - ensrc_cool    -   ensrc_sink1 - ensrc_sink2 )

	rhores_L2. append( np.sqrt(np.average(( rhores[-1][l:r,l:r]*buffer_mask)**2)) )
	momxres_L2.append( np.sqrt(np.average((momxres[-1][l:r,l:r]*buffer_mask)**2)) )
	momyres_L2.append( np.sqrt(np.average((momyres[-1][l:r,l:r]*buffer_mask)**2)) )
	enres_L2.  append( np.sqrt(np.average((  enres[-1][l:r,l:r]*buffer_mask)**2)) )

rhores_L2  = np.array( rhores_L2)
momxres_L2 = np.array(momxres_L2)
momyres_L2 = np.array(momyres_L2)
enres_L2   = np.array(  enres_L2)
t          = np.array(t)
