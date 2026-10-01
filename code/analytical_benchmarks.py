# Compute symbolic identities, initialization constants and scalar-model benchmarks.
from __future__ import annotations
import csv
import json
from pathlib import Path
import mpmath as mp
import sympy as sp
from scipy.integrate import quad
from scipy.special import hyp2f1

HERE = Path(__file__).resolve().parent
mp.mp.dps = 100

def symbolic_checks() -> dict[str, bool]:
    d, a, de, q = sp.symbols('d a de q', positive=True)
    ga = 1-de
    powers = [1/(1-q)]
    for _ in range(4):
        powers.append(sp.simplify(q*sp.diff(powers[-1], q)))
    S = [v.subs(q, ga**2) for v in powers]
    b0 = de**2*S[0]
    b1 = (ga**2*S[0]-2*ga*de*S[1]+de**2*S[2])/ga**2
    b2 = (de**2*S[4]-2*de*(de+2*ga)*S[3]+(de+2*ga)**2*S[2])/ga**4
    s = d/(d+2)
    c = d/(d+a)
    Q = (3*d+8-a)/(4*(d+2))
    A = a*s/4
    mu = a*s/8 + a*(a-2)*(a+d-2)*s/(32*(d+2))
    nu = mu-a*a*s/4+a*(a-1)*c/2
    chi = A*Q+nu
    b4 = de**4/(1-ga**4)
    m4 = s*s*(1+2/d)*b0*b0 + (d/(d+4)-s*s*(1+2/d))*b4
    tests = {
        'coefficient_l2': sp.simplify(b0-de/(2-de)) == 0,
        'coefficient_first_derivative_l2': sp.simplify(b1-2/(de*(2-de)**3)) == 0,
        'coefficient_second_derivative_l2': sp.simplify(b2-8*(1-de+de**2)/(de**3*(2-de)**5)) == 0,
        'fourth_state_moment_leading': sp.simplify(sp.limit(m4/de**2, de, 0)-s/4) == 0,
        'chi_at_one': sp.simplify(chi.subs(a,1)-d*(d+7)/(32*(d+2)**2)) == 0,
        'q_identity': sp.simplify(Q-(1-s/(4*c))) == 0,
    }
    r = sp.symbols('r', positive=True)
    H11 = 2*sp.integrate((1-r)/r*(r*r/2),(r,0,1))
    tests['H_1_1_exact'] = H11 == sp.Rational(1,6)
    assert all(tests.values()), tests
    return tests

def H_numeric(d: int, alpha: float = 1.0) -> float:


    if d < 1 or alpha <= 0:
        raise ValueError('d >= 1 and alpha > 0 required')
    c = d/(d+alpha)
    def integrand(r: float) -> float:
        if r == 0:
            return 0.0
        if r < .001:
            gap = alpha*r*r/2 + alpha*(alpha-2)*(d+alpha-2)*r**4/(8*(d+2))
        else:
            gap = c*(hyp2f1(-alpha/2,-(d+alpha)/2,d/2,r*r)-1)
        return 2*(1-r**d)*gap/r
    val, err = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12, limit=300)
    if err > 2e-10:
        raise ArithmeticError(f'Integral uncertainty for d={d}: {err}')
    return val

def dimension_checks() -> list[dict]:
    table_H=[.166667,.244533,.289286,.338504,.387483,.417395,.437521,.444638]
    rows=[]
    for d, reported in zip([1,2,3,5,10,20,50,100],table_H):
        H=H_numeric(d)
        la=2*mp.log(mp.mpf(4)*(d+2)/(3*d+7))
        c=mp.mpf(d)/(d+1);q=mp.mpf(3*d+7)/(4*(d+2))
        pref=mp.sqrt(H/(c*q*la))
        rows.append(dict(d=d,H=H,lambda_star=float(la),critical_prefactor=float(pref),
                         reported_H=reported,within_rounding=abs(H-reported)<.50001e-6))
    assert all(x['within_rounding'] for x in rows),rows
    return rows

def finite_1d_excess(delta: mp.mpf, N: int) -> mp.mpf:

    if not 0 < delta < 1:
        raise ValueError('0 < delta < 1 required')
    tail = -mp.expm1(2*N*mp.log1p(-delta))
    return N*delta/(6*(2-delta)) + (1-delta)*tail/(3*delta*(2-delta)**2)

def exact_finite_checks() -> list[dict]:
    rows=[]
    for N in [16,256,4096,65536,1048576]:
        guess=1/mp.sqrt(N)
        root=mp.findroot(lambda x:mp.diff(lambda y:finite_1d_excess(y,N),x),(.7*guess,1.2*guess))
        slope=mp.diff(lambda x:finite_1d_excess(x,N),root)
        curvature=mp.diff(lambda x:finite_1d_excess(x,N),root,2)
        assert 0 < root < 1 and abs(slope)<mp.mpf('1e-65') and curvature>0

        grid=[mp.exp(mp.log(mp.mpf('0.0000001'))*(1-mp.mpf(j)/299)) for j in range(299)]
        value=finite_1d_excess(root,N)
        assert min(finite_1d_excess(x,N) for x in grid) >= value-mp.mpf('1e-60')
        rows.append(dict(N=N,delta=mp.nstr(root,30),sqrtN_delta=mp.nstr(mp.sqrt(N)*root,20),
                         six_scaled_excess=mp.nstr(6*value/mp.sqrt(N),20),
                         absolute_slope=mp.nstr(abs(slope),6),curvature_positive=bool(curvature>0)))
    return rows

def critical_scalar_checks() -> list[dict]:


    d=2; la=2*mp.log(mp.mpf(16)/13)
    rows=[]
    for L in [64,256,1024,4096]:
        for path,shift in [('centered',mp.mpf(0)),('below_weak',-1/mp.sqrt(L)),('above_weak',1/mp.sqrt(L))]:
            eps=(la+shift)/L
            alpha=1+eps;c=mp.mpf(d)/(d+alpha);q=(3*d+8-alpha)/(4*(d+2))
            H=mp.mpf(str(H_numeric(d,float(alpha))))
            logd=mp.log(q)/eps
            logarg=mp.log(H/(c*q*eps))-L-2*logd
            w=mp.lambertw(mp.exp(logarg))
            target=mp.log(H/(2*alpha*c*q))-L-2*logd
            def equation(t:mp.mpf)->mp.mpf:
                u=mp.exp(t)
                return 2*u+mp.log(mp.expm1(eps*u))-target
            lo=mp.log(w/2)-2;hi=mp.log(w/2)+2
            while equation(lo)>0:lo-=2
            while equation(hi)<0:hi+=2
            for _ in range(340):
                mid=(lo+hi)/2
                if equation(mid)>0:hi=mid
                else:lo=mid
            u=mp.exp((lo+hi)/2)
            ratio=mp.exp(u-w/2)
            rows.append(dict(logN=L,path=path,epsilon=mp.nstr(eps,18),
                             w=mp.nstr(w,20),model_root_over_Delta=mp.nstr(ratio,20),
                             w_over_sqrtL_div_lambda=mp.nstr(w/(mp.sqrt(L)/la),20)))
    return rows

def main() -> None:
    result={'scope':'Symbolic identities, initialization constants, the exact one-dimensional finite objective at power one, and roots of the scalar critical model.',
            'symbolic':symbolic_checks(),'dimension_constants':dimension_checks(),
            'exact_1d_finite':exact_finite_checks(),'critical_scalar_model':critical_scalar_checks()}
    (HERE/'check_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for key in ['dimension_constants','exact_1d_finite','critical_scalar_model']:
        rows=result[key]
        with (HERE/(key+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'symbolic':result['symbolic'],'dimension_constants':result['dimension_constants'],'exact_1d_finite':result['exact_1d_finite']},indent=2))
    print('Wrote check_results.json and three CSV tables.')
if __name__ == '__main__':
    main()
