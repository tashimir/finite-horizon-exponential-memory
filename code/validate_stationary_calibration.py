# Validate disk memory costs and calibration gains with exact formulas and independent high-precision moment calculations.
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import mpmath as mp
import numpy as np

sys.dont_write_bytecode = True
solver = None


def number(value):
    if isinstance(value, np.floating):
        return mp.mpf(np.format_float_scientific(value, precision=35, unique=False))
    return mp.mpf(float(value))


def coeffs(a, order):
    result = [2/(2+a)]
    for k in range(order):
        result.append(result[-1]*(k-a/2)*(k-1-a/2)/(k+1)**2)
    return result


def matrix_rows(d, order):
    lg = mp.log1p(-d)
    return [[mp.mpf(math.comb(k, j)**2)*mp.exp(2*j*lg)*d**(2*(k-j))/(k-j+1)
             for j in range(k+1)] for k in range(order+1)]


def spectral_average(d, N, order):
    if d == 0 or d == 1:
        return [mp.mpf(1)/(k+1) for k in range(order+1)]
    lg = mp.log1p(-d)
    rows = matrix_rows(d, order)
    amplitudes = [[mp.mpf(1)]]
    factors = [mp.mpf(1)] + [-mp.expm1(2*j*N*lg)/(-N*mp.expm1(2*j*lg))
                              for j in range(1, order+1)]
    for k in range(1, order+1):
        terms = []
        for j in range(k):
            denominator = mp.exp(2*j*lg)*(-mp.expm1(2*(k-j)*lg))
            terms.append(mp.fsum(rows[k][ell]*amplitudes[ell][j]
                                 for ell in range(j, k))/denominator)
        terms.append(mp.mpf(1)/(k+1)-mp.fsum(terms))
        amplitudes.append(terms)
    return [mp.fsum(a*f for a, f in zip(row, factors)) for row in amplitudes]


def direct_average(d, N, order):
    rows = matrix_rows(d, order)
    state = [mp.mpf(1)/(k+1) for k in range(order+1)]
    total = [mp.mpf(0)]*(order+1)
    for _ in range(N):
        total = [t+s for t, s in zip(total, state)]
        state = [mp.fsum(t*s for t, s in zip(row, state)) for row in rows]
    return [x/N for x in total]


def independent_average(d, N, order):
    if d == 0 or d == 1:
        return [mp.mpf(1)/(k+1) for k in range(order+1)]
    if N <= 512:
        return direct_average(d, N, order)
    if -N*mp.log1p(-d) <= 250:
        return spectral_average(d, N, order)
    rows = matrix_rows(d, order)
    stationary = [mp.mpf(1)]
    correction = [mp.mpf(0)]
    remainder = [mp.mpf(0)]
    decay = mp.exp(N*mp.log1p(-d))
    for k in range(1, order+1):
        denominator = -mp.expm1(2*k*mp.log1p(-d))
        stationary.append(mp.fsum(rows[k][j]*stationary[j] for j in range(k))/denominator)
        correction.append((mp.mpf(1)/(k+1)-stationary[k]
                           + mp.fsum(rows[k][j]*correction[j] for j in range(k)))/denominator)
        remainder.append((4*k*decay+mp.fsum(rows[k][j]*remainder[j] for j in range(k)))/denominator)
    assert max(remainder)/N < mp.mpf('1e-85')
    return [s+u/N for s,u in zip(stationary, correction)]


def stationary_with_derivative(d, order):
    lg = mp.log1p(-d)
    rows = matrix_rows(d, order)
    moments = [mp.mpf(1)]
    derivatives = [mp.mpf(0)]
    for k in range(1, order+1):
        den = -mp.expm1(2*k*lg)
        value = mp.fsum(rows[k][j]*moments[j] for j in range(k))/den
        numerator_prime = mp.fsum(rows[k][j]*(derivatives[j]
            + moments[j]*(2*(k-j)/d-2*j/(1-d))) for j in range(k))
        derivative = (numerator_prime-2*k*mp.exp((2*k-1)*lg)*value)/den
        moments.append(value)
        derivatives.append(derivative)
    return moments, derivatives


def stationary_gradient(x, a, cs):
    d = mp.exp(x)
    m, dm = stationary_with_derivative(d, len(cs)-1)
    h = (1-d)**a+d**a
    hp = a*(d**(a-1)-(1-d)**(a-1))
    return hp*mp.fsum(c*v for c, v in zip(cs, m))+h*mp.fsum(c*v for c, v in zip(cs, dm))


def excess(a, d, N, order):
    cs = coeffs(a, order)
    m = independent_average(d, N, order+1)
    hm1 = mp.mpf(0) if d == 0 or d == 1 else mp.expm1(a*mp.log1p(-d))+d**a
    e = cs[0]*hm1+(1+hm1)*mp.fsum(c*v for c, v in zip(cs[1:], m[1:]))
    return e, m[-1]


def refined_parameters(a, N, stationary_start, order):
    cs_H = coeffs(a, 512)
    H = mp.fsum(cs_H[k]/(k*(k+1)) for k in range(1, len(cs_H)))
    eps = a-1
    q = (14-a)/16
    c = 2/(2+a)
    lower = mp.log(q)/eps
    target = mp.log(H/(2*N*a*c))
    def balance(x):
        return 2*x+mp.log(q)+mp.log(mp.expm1(eps*(x-lower)))-target
    lo, hi = lower, mp.mpf(0)
    for _ in range(300):
        mid = (lo+hi)/2
        if balance(mid) > 0:
            hi = mid
        else:
            lo = mid
    xf = (lo+hi)/2
    cs = coeffs(a, order)
    xs = mp.log(stationary_start)
    xs = mp.findroot(lambda x: stationary_gradient(x, a, cs),
                     (xs-mp.mpf('.001'), xs+mp.mpf('.001')),
                     tol=mp.mpf('1e-65'), maxsteps=50)
    return mp.exp(xf), mp.exp(lower), mp.exp(xs)


def text(value):
    return mp.nstr(value, 35)


def main():
    global solver
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=Path, default=Path(__file__).with_name('stationary_calibration_comparison.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--precision', type=int, default=100)
    args = parser.parse_args()
    if args.precision < 90:
        parser.error('At least 90 decimal digits are required.')
    mp.mp.dps = args.precision
    spec = importlib.util.spec_from_file_location('disk_comparison', args.solver)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    references = []
    def record(case_id, quantity, observed, reference, tolerance, method, **parameters):
        difference = abs(number(observed)-reference) if isinstance(observed, (float,np.floating)) else abs(observed-reference)
        rows.append(dict(case_id=case_id, quantity=quantity, N=parameters.get('N',''),
                    alpha=parameters.get('alpha',''), delta=parameters.get('delta',''),
                    order=parameters.get('order',''), calculated_value=str(observed),
                    reference_value=text(reference), absolute_difference=text(difference),
                    tolerance=text(tolerance), passed=bool(difference <= tolerance),
                    reference_method=method))
    for power in [8,20,32,40]:
        N=2**power
        for index,d0 in enumerate([.926/N,N**(-1/3),.2,.7]):
            d=number(d0)
            observed=solver.finite_moments(d0,N,65)
            exact=d/(2*(2-d))+(1-d)*(-mp.expm1(2*N*mp.log1p(-d)))/(N*d*(2-d)**2)
            record(f'moment_{power}_{index}','mean_squared_radius',observed[1],exact,mp.mpf('5e-15'),
                   'closed first-moment formula',N=N,delta=d0,order=65)
            observed_e,_=solver.mean_excess(2,d0,N,40)
            exact_e=((1-d)**2+d**2)*(mp.mpf('.5')+exact)-mp.mpf('.5')
            record(f'quadratic_{power}_{index}','mean_cost_excess',observed_e,exact_e,mp.mpf('5e-15'),
                   'closed quadratic-cost formula',N=N,alpha=2,delta=d0,order=40)
    observed=solver.stationary_minimizer(2,40)
    record('quadratic_stationary','stationary_delta',observed,2-mp.sqrt(10)/2,mp.mpf('2e-13'),
           'closed quadratic stationary minimizer',alpha=2,order=40)
    for delta in [0,1]:
        e,b=solver.mean_excess(1,delta,256,96)
        reference=128/(45*mp.pi)-mp.mpf(2)/3
        record(f'linear_endpoint_{delta}','mean_cost_excess',e,reference,number(b)+mp.mpf('2e-14'),
               'mean distance of two independent disk points',N=256,alpha=1,delta=delta,order=96)
    cs,tail=solver.potential(1,512)
    record('linear_potential_boundary','potential',np.sum(cs)-tail,32/(9*mp.pi),mp.mpf('5e-15'),
           'closed disk boundary potential',alpha=1,delta=1,order=512)
    for d0,N,K in [(mp.mpf('.13'),13,12),(mp.mpf('.7'),17,18)]:
        one=spectral_average(d0,N,K)
        two=direct_average(d0,N,K)
        difference=max(abs(a-b) for a,b in zip(one,two))
        record(f'spectral_{N}','maximum_moment_difference',difference,mp.mpf(0),mp.mpf('1e-70'),
               'direct high-precision recurrence',N=N,delta=text(d0),order=K)
    for alpha in [1,1.2,1.5,1.8]:
        a=number(alpha)
        def integrand(r):
            if r==0:
                return mp.mpf(0)
            return 2*(1-r*r)*(2/(2+a)*mp.hyp2f1(-a/2,-1-a/2,1,r*r)-2/(2+a))/r
        reference=mp.quad(integrand,[0,mp.mpf('.5'),mp.mpf('.9'),1])
        observed,bound=solver.initialization_constant(alpha)
        record(f'initialization_{alpha}','H',observed,reference,number(bound)+mp.mpf('2e-15'),
               'high-precision quadrature of the disk potential',alpha=alpha,order=512)
    if args.cases:
        with args.cases.open(encoding='utf-8-sig',newline='') as stream:
            cases=list(csv.DictReader(stream))
    else:
        cases=[]
        for power,kind,value in [(40,'r',.1),(40,'r',.5),(40,'r',1.5),(40,'r',1.55),
                                 (40,'r',3),(8,'r',1.5),(8,'alpha',1.8),(40,'alpha',1.8)]:
            N=2**power
            alpha=1+value*solver.LAMBDA_STAR/math.log(N) if kind=='r' else value
            cases.append(dict(case_id=f'gain_{power}_{kind}_{value}',N=N,alpha=alpha,order=''))
    for item in cases:
        N=int(item['N']); alpha=float(item['alpha']); a=number(alpha)
        values=(solver.evaluate(alpha,N,int(item['order'])) if item.get('order') else solver.scenario(alpha,N))
        K=int(values['order'])
        df,da,ds=[number(values[key]) for key in ['delta_finite','delta_stationary_scale','delta_stationary']]
        if alpha==1:
            cs=coeffs(a,512)
            H=mp.fsum(cs[k]/(k*(k+1)) for k in range(1,len(cs)))
            df2=mp.sqrt(H/(2*N*(2/(2+a))*(1-(14-a)/16)))
            da2=ds2=mp.mpf(0)
        else:
            df2,da2,ds2=refined_parameters(a,N,ds,K)
        ef,mf=excess(a,df2,N,K)
        ea,ma=excess(a,da2,N,K)
        es,ms=excess(a,ds2,N,K)
        c=2/(2+a)
        gs=100*(es-ef)/(c+es)
        ga=100*(ea-ef)/(c+ea)
        case_id=item.get('case_id',f'gain_{N}_{alpha}')
        for name,reference in [('stationary',gs),('scale',ga)]:
            record(case_id,f'saving_vs_{name}_pct',values[f'saving_vs_{name}_pct'],reference,
                   min(mp.mpf('1e-7'),abs(reference)/10),'high-precision moments and stationary derivative root',
                   N=N,alpha=alpha,order=K)
        if ds2:
            record(case_id,'stationary_delta_relative_error',abs(ds-ds2)/ds2,mp.mpf(0),mp.mpf('1e-11'),
                   'high-precision stationary derivative root',N=N,alpha=alpha,order=K)
        references.append(dict(case_id=case_id,N=N,alpha=alpha,order=K,delta_finite=text(df2),
                        delta_stationary_scale=text(da2),delta_stationary=text(ds2),
                        saving_vs_stationary_pct=text(gs),saving_vs_scale_pct=text(ga),
                        finite_averaged_tail_moment=text(mf),stationary_averaged_tail_moment=text(ms),
                        scale_averaged_tail_moment=text(ma)))
        print(json.dumps(dict(case_id=case_id,N=N,alpha=alpha,order=K)),flush=True)
    for filename,items in [('numerical_validation.csv',rows),('high_precision_reference_cases.csv',references)]:
        with (args.output/filename).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(items)
    metadata=dict(decimal_precision=args.precision,solver_sha256=hashlib.sha256(args.solver.read_bytes()).hexdigest(),
                  numpy_longdouble_precision=int(np.finfo(np.longdouble).precision),checks=len(rows),
                  all_passed=all(row['passed'] for row in rows),
                  reference_scope='Exact formulas and independent high-precision calculations on the listed cases. Floating-point interval certification is outside the scope of these checks.')
    (args.output/'numerical_validation.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata),flush=True)
    if not metadata['all_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
