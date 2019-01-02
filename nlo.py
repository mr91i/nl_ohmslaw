import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import math

############################
pi	  = 3.141592653589793
m_e   = 9.10938291e-28
q_e   = -4.803205e-10 
e	  = 4.803205e-10
k_B   = 1.3806488e-16

sigma_en = 1.0e-15
K_in	 = 1.6e-9
G_const  = 6.67e-8
M_Sun	 = 1.988435e33
c_light  = 2.998e10
one_AU	 = 1.496e13
one_amu  = 1.660468e-24
one_year = 3.154e7

m_n		= 3.88549512e-24
m_i		= 4.81556287e-23
rho_d	= 2.0
q_i		= +4.803205e-10
Mass_star = 1.988435e33
L_Sun = 3.846e33
############################
## alias
sqrt = np.sqrt 
exp = np.exp
##
####### 
#beta_center=1e5
#r_d=0.1e-4
#f_dg=1.0e-2
############################

def n_calc( T_n , zeta , rho_gas , f_dg , E , r_d ):
	n_n = rho_gas/m_n
	n_d = 0.75*m_n*f_dg*n_n/(pi*rho_d*r_d*r_d*r_d)
	E_crit = sqrt(6.0*m_e/m_n)*k_B*n_n*sigma_en*T_n/e

	T_e , T_i = get_T_plasma( T_n , E , E_crit )
	Z = Z_calc_by_NR( E , T_e , T_i , n_n , T_n , n_d , r_d , zeta)

	if Z is np.nan:
		n_i = np.nan
		n_e = np.nan

	mft_i = MeanFreeTime_i_func(n_n)
	v_drift_i = (m_n+m_i)/(m_i*m_n) * q_i* E * mft_i
	K_de  = K_de_func(r_d, T_e, e*Z/r_d)
	K_di  = K_di_func(r_d,T_n,T_i,e*Z/r_d,v_drift_i)
	K_rec = K_rec_func(T_e)
 
	n_i = n_i_func(K_de,K_di,K_rec,n_n,n_d,zeta)
	n_e = n_e_func(K_de,K_di,K_rec,n_n,n_d,zeta)

	return n_e , n_i , Z*n_d

def MeanFreeTime_e_func( n_n , T_e ):
	return 3.0/(16.0 * sigma_en * n_n * sqrt( k_B * T_e / ( 2.0 * pi * m_e ) ) )

def MeanFreeTime_i_func( n_n ):
	return 1.0/(K_in*n_n)

def T_e_func(T_n, E , E_crit):
	x = E/E_crit
	return T_n*(0.5 + sqrt(0.25 + 9.0*pi/64.0 *x*x ))

def T_i_func(T_n , E , E_crit):
	x		 = E/E_crit
	mu_in	 = m_i*m_n/(m_i + m_n)
	kappa_in = 2.0* m_i*m_n/(m_i+m_n)/(m_i+m_n)
	return T_n * ( 1.0 +  4.0*m_e*k_B*T_n*sigma_en**2/( mu_in*kappa_in*m_n*K_in**2 ) *x*x)

def get_T_plasma(T_n, E , E_crit ):
	x = E/E_crit
	mu_in = m_i*m_n/(m_i + m_n)
	kappa_in = 2.0* m_i*m_n/(m_i+m_n)/(m_i+m_n)
	T_e = T_n*(0.5 + sqrt(0.25 + 9.0*pi/64.0 *x*x ))
	T_i = T_n * ( 1.0 + 4.0*m_e*k_B*T_n*sigma_en**2/(mu_in*kappa_in*m_n*K_in**2) *x*x)
#	T_e = T_n * (0.5 + sqrt(0.25 + 2/3*x*x ))
#	T_i = T_n*( 1 + 7.6e-7*(T_n/100)*x*x )
	return T_e, T_i

def drift_velocity_e_func(n_n,T_e,E):
	mfp = 1.0/n_n/sigma_en
	return - 3.0 * sqrt( 3.0 * pi ) * e * mfp * E /( 16.0 * sqrt( m_e * 1.5*k_B*T_e ) )

def drift_velocity_i_func(n_n,T_e,E):
	mft_i = MeanFreeTime_i_func( n_n )
	return (m_n+m_i)/(m_i*m_n)*q_i*E*mft_i

def K_de_func(r_d,T_e,phi_d):
	Psi_e = -e*phi_d/k_B/T_e

	if( Psi_e < 0.0 ):
		K_de = pi*r_d**2 * sqrt( 8.0 *k_B*T_e  /pi/m_e	)  *  ( 1.0 - Psi_e )
	else:
		K_de = pi*r_d**2 * sqrt( 8.0 *k_B*T_e  /pi/m_e	)  *  exp( -Psi_e )
	if(K_de==0.0): return np.nan	 

	return K_de

def K_di_func(r_d,T_n,T_i,phi_d,v_drift):
	K_di	  = 0.0
	theta	  = T_i/T_n
	u		  = v_drift/sqrt(k_B*T_n/m_i)
	Psi		  = -q_i*phi_d/k_B/T_n
	u_thermal = sqrt(8.0*k_B*T_n/pi/m_i)
	if(phi_d < 0.0 ):
#		K_di = pi*r_d**2*u_thermal * (0.5*sqrt(theta)*exp(-0.5*u**2/theta) + sqrt(pi/8.0)*( theta + 2.0*Psi + u**2 )/u * special.erf( u/sqrt(2.0*theta) ) )
		K_di = pi*r_d**2*( sqrt(2*k_B*T_i/pi/m_i) *exp(-m_i*v_drift*v_drift/2/k_B/T_i) + abs(v_drift) * ( 1+(k_B*T_i+2*e*abs(phi_d)/m_i/v_drift**2 ) )* special.erf( abs(v_drift)/sqrt(2*k_B*T_i/m_i) ) )
	else:
		K_di = 4.0*pi*r_d**2 *sqrt( k_B*T_i/(2.0*pi*m_i) )*exp( -q_i*phi_d /k_B/T_i )
	if(K_di==0.0): return np.nan	 
	return K_di


def K_rec_func(T_e):
	return 2.4e-7*(T_e/300.0)**(-0.69)


def n_e_func(K_de,K_di,K_rec,n_n,n_d,zeta):
	A = K_di*K_de*n_d*n_d/K_rec /zeta /n_n
	return zeta*n_n/K_de/n_d /(0.5+sqrt(0.25+1.0/A))


def n_i_func(K_de,K_di,K_rec,n_n,n_d,zeta):
	A = K_di*K_de*n_d*n_d/K_rec /zeta /n_n
	return zeta*n_n/K_di/n_d /(0.5+sqrt(0.25+1.0/A))

def get_n_plasma( K_de,K_di,K_rec ,n_n,n_d,zeta ):
	A	= K_di*K_de*n_d*n_d/K_rec /zeta /n_n
	n_e = zeta*n_n/K_de/n_d /(0.5+sqrt(0.25+1.0/A))
	n_i = zeta*n_n/K_di/n_d /(0.5+sqrt(0.25+1.0/A))
	return n_e , n_i

def Z_new_func( n_i, n_e, n_d):
	return (n_e-n_i)/n_d

def E_crit_func( T, n_n):
	return 1.0e-9*(T/100.0)*(n_n/1.0e12)

def sigma_0( T , n_n, n_e , n_i , Znd ):
    E = 1e-30
    j_e = -e * n_e * drift_velocity_e_func( n_n , T , E )
    j_i =  e * n_i * drift_velocity_i_func( n_n , T , E )
    return j_e/E , j_i/E

def sigma( T , n_n, n_e , n_i , Znd , E):
	j_e = -e * n_e * drift_velocity_e_func( n_n , T , E )
	j_i =  e * n_i * drift_velocity_i_func( n_n , T , E )
	return j_e/E , j_i/E

def J_EH( T , n_n , sigma_e ):
    E_EH = E_crit_func( T, n_n )
    return sigma_e * E_EH

def J_max( r , n_n , f=10):
	 return 0.0471828858 * (f/10) * np.sqrt(n_n/1e12) * r**(-1.5)

#def Z_calc_by_NR( E, T_e, T_i, m_n, n_n, T_n, m_i, q_i, n_d, r_d, zeta):
def Z_calc_by_NR( E, T_e, T_i, n_n, T_n,  n_d, r_d, zeta):
	Z = 0.0
		
	Z_0		  = -2.0*(T_n/100.0)
	delta	  = 1.0e-3*Z_0
	Z_start   = Z_0
	mft_i	  = MeanFreeTime_i_func(n_n)
	v_drift_i = (m_n+m_i)/(m_i*m_n) *q_i*E*mft_i
	K_rec	  = K_rec_func(T_e)
	i=0
	while True:
		i+=1
		
		n_e , n_i = get_n_plasma( K_de_func(r_d, T_e, e*Z_0/r_d),  K_di_func(r_d, T_n,T_i,e*Z_0/r_d,v_drift_i)				, K_rec,n_n,n_d,zeta )
		F_0 = Z_0*n_d+n_i-n_e
		
		n_e , n_i = get_n_plasma( K_de_func(r_d, T_e, e*(Z_0+delta)/r_d),  K_di_func(r_d, T_n,T_i,e*(Z_0+delta)/r_d,v_drift_i) , K_rec, n_n, n_d, zeta )
		F_d = (Z_0+delta)*n_d+n_i-n_e
		
		Z = Z_0 - delta*F_0/(F_d - F_0)
		
		if( abs(Z-Z_0) < 1e-8*abs(Z_0) ): break
		if( abs(Z-Z_0) > 1e20*abs(Z_0) ): return np.nan
		
		Z_0=Z
		if( abs(Z) < 1e-100 ):
			Z_start *= 1.1
			Z = Z_start
			Z_0 = Z*0.1
	
	return Z


#def sigma_conductivity_func( T_n, zeta, rho_gas, f_dg, E, r_d):
def J_func( T_n, zeta, rho_gas, f_dg, E, r_d):
	if E < 1.0e-50:
		E = 1.0e-50
	n_n    = rho_gas/m_n
	n_d    = 0.75*rho_gas*f_dg/(pi*rho_d*r_d**3)
	E_crit = sqrt(6.0*m_e/m_n)*k_B*n_n*sigma_en*T_n/e
	
	T_e , T_i = get_T_plasma( T_n, E , E_crit )	
	Z = Z_calc_by_NR( E, T_e, T_i, n_n, T_n,  n_d, r_d, zeta)
	if math.isnan(Z) :
		return np.nan
	
	mft_i = MeanFreeTime_i_func(n_n)

	v_drift_e = drift_velocity_e_func(n_n, T_e, E)
	v_drift_i = (m_n+m_i)/(m_i*m_n)*q_i*E*mft_i

	K_de  = K_de_func(	r_d , T_e , e*Z/r_d )
	K_di  = K_di_func(	r_d , T_n , T_i , e*Z/r_d , v_drift_i )
	K_rec = K_rec_func( T_e )

	n_i = n_i_func(K_de,K_di,K_rec,n_n,n_d,zeta)
	n_e = n_e_func(K_de,K_di,K_rec,n_n,n_d,zeta)   
	
	J_tot = -e * n_e * v_drift_e  +  q_i * n_i * v_drift_i 
	
	return J_tot ,	-e * n_e * v_drift_e ,	q_i * n_i * v_drift_i

def sat_J( r, T_n , zeta , rho_gas , f_dg , r_d , beta ):
	n_n    = rho_gas/m_n
	Jmax = J_max( r , n_n )
	E_0 = 0.01*E_crit_func( T_n, n_n)
	n_e, n_i, Znd = n_calc( T_n , zeta , rho_gas , f_dg , E_0 , r_d )
	sig_e , sig_i = sigma_0( T_n , n_n, n_e , n_i , Znd )
	cs = 1e5*(T_n/280)**(-0.5)
	Omg = 2e-7*r**(-1.5)
	eta = c_light**2/(4*pi*sig_e*cs**2/Omg)
	#eta = 0.390e2*(beta/1e4)**(-0.5) *cs*cs/Omg;
	eta_crit = 0.390e-2*(beta/1e4)**(-0.5) 
	
	#Els = 2/beta/eta
	#Els = 2/beta/eta_crit
	
#	print(Els)
#	print(eta, eta_crit)
	
	#if Els < 1 :
	if eta_crit < eta :
		Els = 2/beta/eta
		return Jmax*Els**(0.5)
	J_0 = 1e30
	E=E_0
	while 1:
		E*=1.2
		#		print(f"E is {E}")
		J, Je, Ji = J_func( T_n, zeta, rho_gas, f_dg, E, r_d)
		if J > Jmax:
			return Jmax
		if J < J_0:
			return J
		J_0 = J
		if E > 1e6*E_0:
			print("Error during calculation of sat J")
			return None


if __name__ == "__main__":
	print("Doing Test : model C in Okuzumi & Inutsuka 2015")
	## default parameter
	T_n  = 100
	zeta = 1e-17
	n_n  = 1e12
	rho  = n_n*m_n #1e-19
	f_dg = 1e-2
	E	 = 1e-50
	r_d  = 1e-4

	n_e , n_i , Znd = n_calc( T_n , zeta , rho , f_dg , E , r_d )
	K_de = K_de_func(r_d , T_n , -3*k_B*T_n/e ) 
	v_e = sqrt(8 * k_B *T_n/pi/m_e)
	E_EH = E_crit_func( T_n, n_n )

	m_d = 4*pi/3 * r_d**3 * rho_d
#	n_d = rho*f_dg/m_d
	print("Coulomb parameter C (E << E_crit) = ", K_de/(pi*r_d**2*v_e))
	print("Plasma abundance x_e x_i -Z*x_d (E << E_crit) = ", n_e/n_n,n_i/n_n, -Znd/n_n)



	Elist = np.logspace(-2,3,100) * E_EH
	
	T_list=[]
	x_list=[]
	J_list=[]
	for E in Elist:
		T_e , T_i = get_T_plasma( T_n, E , E_crit_func( T_n, n_n) )
		n_e , n_i , Znd = n_calc( T_n , zeta , rho , f_dg , E , r_d )
		J , Je , Ji = J_func( T_n, zeta, rho , f_dg, E, r_d)
		T_list.append(	[T_e, T_i] )
		x_list.append(	[ n_e/n_n ,  n_i/n_n , -Znd/n_n ] )
		J_list.append( [J , Je ,Ji ] )
	T_list = np.array( T_list )
	x_list = np.array( x_list )
	J_list = np.array( J_list )


	print("Calc. End")

	plt.figure()
	plt.plot(  Elist , T_list[:,1]	, marker="None" , c="r")
	plt.plot(  Elist , T_list[:,0]	, marker="None" , c="b" )
	plt.xlabel(r"Erictiric Field Strength $E$ [esu cm$^{-2}$]")
	plt.ylabel(r"Plasma Temperature $T_e, T_i$ [K]")
	plt.xscale("log")
	plt.yscale("log")
	plt.savefig("Temperature_C.pdf" , transparent=True)
	plt.close()

	plt.figure()
	for i in range(3):
		plt.plot(  Elist , x_list[:,i]	 , marker="None" )
	plt.xlabel(r"Erictiric Field Strength $E$ [esu cm$^{-2}$]")
	plt.ylabel(r"Abundance $x_e$, $x_i$, $-Zx_d$")
	plt.xscale("log")
	plt.yscale("log")
	plt.savefig("Abundance_C.pdf" , transparent=True)
	plt.close()

	plt.figure()
	plt.plot(  Elist , J_list[:,1]	, marker="None" , c="b")
	plt.plot(  Elist , J_list[:,2]	, marker="None" , c="r")
	plt.plot(  Elist , J_list[:,0]	, marker="None" , c="k" )
	plt.xlabel(r"Erictiric Field Strength $E$ [esu cm$^{-2}$]")
	plt.ylabel(r"Current Density $J$ [esu cm$^{-2}$ s$^{-1}$]")
	plt.xscale("log")
	plt.yscale("log")
	plt.savefig("Current_C.pdf" , transparent=True)
	plt.close()

	print("Plot End")
