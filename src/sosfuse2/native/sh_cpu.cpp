// Derived from results/sos_full/b2_run/sh_cpu.cpp; runtime H, g and thread count.

#include <cmath>
#include <algorithm>
#include <vector>
#include <thread>
#include <cstdint>
#ifndef RIG_THREADS
#define RIG_THREADS 2
#endif
static constexpr double PI=3.14159265358979323846;
static int ix(int l,int m){return l*(l+1)/2+m;}
struct Norm {
 int L,K; std::vector<double>a,b,d;
 Norm(int ll):L(ll),K((ll+1)*(ll+2)/2),a(K),b(K),d(ll+1){
  for(int m=1;m<=L;m++)d[m]=-sqrt((2.*m+1)/(2*m));
  for(int l=1;l<=L;l++)for(int m=0;m<l;m++){
   a[ix(l,m)]=sqrt((4.*l*l-1)/(l*l-m*m));
   if(l>m+1)b[ix(l,m)]=sqrt((2.*l+1)*((l-1.)*(l-1)-m*m)/((2.*l-3)*(l*l-m*m)));
  }
 }
 void eval(double mu,double cp,double*y)const {
  mu=std::clamp(mu,-1.,1.);cp=std::clamp(cp,-1.,1.);
  double st=sqrt(std::max(0.,1-mu*mu)),diag=1/sqrt(4*PI),cm=1,prev=cp;
  for(int m=0;m<=L;m++){
   if(m==1){diag*=d[m]*st;cm=cp;prev=1;}
   else if(m>1){diag*=d[m]*st;double next=2*cp*cm-prev;prev=cm;cm=next;}
   double mult=m?std::sqrt(2.)*cm:1.;
   double z0=diag;y[ix(m,m)]=mult*z0;
   if(m<L){double z1=sqrt(2.*m+3)*mu*z0;y[ix(m+1,m)]=mult*z1;
    for(int l=m+2;l<=L;l++){int k=ix(l,m);double z2=a[k]*mu*z1-b[k]*z0;y[k]=mult*z2;z0=z1;z1=z2;}
   }
  }
 }
};
extern "C" void basis(int n,int L,const double*dirs,double*out){
 Norm norm(L);for(int i=0;i<n;i++){
  double mu=dirs[3*i+2],st=sqrt(std::max(0.,1-mu*mu));
  norm.eval(mu,st>1e-12?dirs[3*i]/st:1.,out+(int64_t)i*norm.K);
 }
}
static double hg(double u,double g){double d=std::max(1e-20,1+g*g-2*g*u);return (1-g*g)/(4*PI*d*sqrt(d));}
struct Args {
 int nr,nw,L,nchi,ns; const int64_t*off;const int32_t*base;const uint8_t*lit;
 const float*p;const double*nu;const float*src;const double*ext;const double*g;
 const int64_t*goff;const float*gdir;const float*gval;const double*scatter;
 double*out;int first;
};
template<int W> static void runpart(const Args&A,int lo,int hi,const Norm&norm){
 const int K=norm.K,M=A.L+1; std::vector<double>Y(K);std::vector<int>stamp(A.ns,-1);
 std::vector<float>cache((int64_t)A.ns*M*W);
 for(int ray=lo;ray<hi;ray++){
  double sum[W]={},pr=3*(1+A.nu[ray]*A.nu[ray])/(16*PI),pa[W];
  for(int w=0;w<W;w++)pa[w]=hg(A.nu[ray],A.g[w]);
  if(A.first!=1)norm.eval(A.nu[ray],1.,Y.data());
  for(int64_t n=A.off[ray];n<A.off[ray+1];n++){
   const float*q=A.p+13*n;double ah=q[0],ac=q[1],mu=q[2],cp=q[3],ds=q[4],h=q[5];
   int cells[4]={A.base[n],A.base[n]+1,A.base[n]+A.nchi,A.base[n]+A.nchi+1};
   double sw[4]={(1-ah)*(1-ac),(1-ah)*ac,ah*(1-ac),ah*ac};
   double val[W]={};
   if(A.first==1){
    double rr=exp(-std::max(0.,h)/8),aa=exp(-std::max(0.,h)/1.2);
    for(int w=0;w<W;w++){
     double ts=A.lit[n]?exp(-q[9]*A.ext[w]-q[10]*A.ext[W+w]-q[11]*A.ext[2*W+w]):0;
     val[w]=ts*(rr*A.ext[w]*pr+aa*A.scatter[w]*pa[w]);
    }
    double st=sqrt(std::max(0.,1-mu*mu)),sp=sqrt(std::max(0.,1-cp*cp));
    for(int c=0;c<4;c++)if(sw[c]>1e-12){
     for(int64_t j=A.goff[cells[c]];j<A.goff[cells[c]+1];j++){
      double x=A.gdir[3*j],y=A.gdir[3*j+1],z=A.gdir[3*j+2];
      double u0=mu*z+st*cp*x,u1=st*sp*y,up=std::clamp(u0+u1,-1.,1.),um=std::clamp(u0-u1,-1.,1.);
      double rayp=3*(2+up*up+um*um)/(32*PI);
      for(int w=0;w<W;w++)val[w]+=sw[c]*A.gval[j*W+w]*(rr*A.ext[w]*rayp+aa*A.scatter[w]*.5*(hg(up,A.g[w])+hg(um,A.g[w])));
     }
    }
   } else if(A.first==2) {
    norm.eval(A.nu[ray],q[12],Y.data());
    const float*c0=A.src+(int64_t)cells[0]*K*W,*c1=A.src+(int64_t)cells[1]*K*W;
    const float*c2=A.src+(int64_t)cells[2]*K*W,*c3=A.src+(int64_t)cells[3]*K*W;
    for(int k=0;k<K;k++)for(int w=0;w<W;w++){
     int t=k*W+w;val[w]+=Y[k]*(sw[0]*c0[t]+sw[1]*c1[t]+sw[2]*c2[t]+sw[3]*c3[t]);
    }
   } else {
    for(int c=0;c<4;c++)if(stamp[cells[c]]!=ray){
     float*v=cache.data()+(int64_t)cells[c]*M*W;std::fill(v,v+M*W,0.f);
     const float*s=A.src+(int64_t)cells[c]*K*W;
     for(int l=0;l<=A.L;l++)for(int m=0;m<=l;m++){
      int k=ix(l,m);float pref=(float)Y[k];for(int w=0;w<W;w++)v[m*W+w]+=pref*s[k*W+w];
     }
     stamp[cells[c]]=ray;
    }
    const float*c0=cache.data()+(int64_t)cells[0]*M*W,*c1=cache.data()+(int64_t)cells[1]*M*W;
    const float*c2=cache.data()+(int64_t)cells[2]*M*W,*c3=cache.data()+(int64_t)cells[3]*M*W;
    double cm=1,prev=q[12];
    for(int m=0;m<M;m++){
     if(m==1){cm=q[12];prev=1;}else if(m>1){double next=2*q[12]*cm-prev;prev=cm;cm=next;}
     for(int w=0;w<W;w++){int t=m*W+w;val[w]+=cm*(sw[0]*c0[t]+sw[1]*c1[t]+sw[2]*c2[t]+sw[3]*c3[t]);}
    }
   }
   for(int w=0;w<W;w++)sum[w]+=ds*exp(-q[6]*A.ext[w]-q[7]*A.ext[W+w]-q[8]*A.ext[2*W+w])*val[w];
  }
  for(int w=0;w<W;w++)A.out[(int64_t)ray*W+w]=sum[w];
 }
}
template<int W> static void dispatch(const Args&A){
 Norm norm(A.L);std::vector<std::thread> ts; for(int k=0;k<RIG_THREADS;k++)ts.emplace_back([&,k](){runpart<W>(A,A.nr*k/RIG_THREADS,A.nr*(k+1)/RIG_THREADS,norm);});for(auto &t:ts)t.join();
}
extern "C" void transport(int nr,int nw,int L,int nchi,int ns,const int64_t*off,const int32_t*base,
 const uint8_t*lit,const float*p,const double*nu,const float*src,const double*ext,const double*g,
 const int64_t*goff,const float*gdir,const float*gval,const double*scatter,double*out,int first){
 Args A{nr,nw,L,nchi,ns,off,base,lit,p,nu,src,ext,g,goff,gdir,gval,scatter,out,first};
 switch(nw){case 1:dispatch<1>(A);break;case 4:dispatch<4>(A);break;case 15:dispatch<15>(A);break;}
}

static void compactpart(int nr,int L,int nchi,int ns,int density,const int64_t*off,const int32_t*base,
 const float*p,const double*nu,const float*src,double*out,int lo,int hi){
 Norm norm(L);int K=norm.K,M=L+1,S=K+6*density,C=M+3*density,P=4+2*density;std::vector<double>Y(K);
 std::vector<int>stamp(ns,-1);std::vector<float>cache((int64_t)ns*C);
 for(int ray=lo;ray<hi;ray++){
  norm.eval(nu[ray],1.,Y.data());double sum=0;
  for(int64_t n=off[ray];n<off[ray+1];n++){
   const float*q=p+P*n;double ah=q[0],ac=q[1],cp=q[2],wt=q[3];
   int cells[4]={base[n],base[n]+1,base[n]+nchi,base[n]+nchi+1};
   double sw[4]={(1-ah)*(1-ac),(1-ah)*ac,ah*(1-ac),ah*ac};
   for(int c=0;c<4;c++)if(stamp[cells[c]]!=ray){
    float*v=cache.data()+(int64_t)cells[c]*C;std::fill(v,v+C,0.f);
    const float*s=src+(int64_t)cells[c]*S;
    if(density)for(int l=0;l<=2;l++)for(int m=0;m<=l;m++){int k=ix(l,m);v[M+m]+=(float)Y[k]*s[k];}
    for(int l=0;l<=L;l++)for(int m=0;m<=l;m++){int k=ix(l,m);v[m]+=(float)Y[k]*s[k+6*density];}
    stamp[cells[c]]=ray;
   }
   double cm=1,prev=cp,val=0;
   for(int m=0;m<M;m++){
    if(m==1){cm=cp;prev=1;}else if(m>1){double next=2*cp*cm-prev;prev=cm;cm=next;}
    for(int c=0;c<4;c++){
     val+=cm*sw[c]*cache[(int64_t)cells[c]*C+m]*(density?q[5]:1.);
     if(density && m<=2)val+=cm*sw[c]*cache[(int64_t)cells[c]*C+M+m]*q[4];
    }
   }
   sum+=wt*val;
  }
  out[ray]=sum;
 }
}
extern "C" void compact_transport(int nr,int L,int nchi,int ns,int density,const int64_t*off,const int32_t*base,
 const float*p,const double*nu,const float*src,double*out){
 std::vector<std::thread>ts;for(int k=0;k<RIG_THREADS;k++)ts.emplace_back([&,k](){compactpart(nr,L,nchi,ns,density,off,base,p,nu,src,out,nr*k/RIG_THREADS,nr*(k+1)/RIG_THREADS);});for(auto&t:ts)t.join();
}

extern "C" void source_interp(int nr,int L,int nc,int ns,int V,const int*mode,
 const int64_t*off,const int32_t*base,const float*p,const double*nu,const float*src,
 const double*hr,const double*ha,const double*poly,const int*left,double*out){
 auto worker=[&](int lo,int hi){
  Norm norm(L);int K=norm.K,M=L+1,S=K+6,C=M+3;
  std::vector<double>Y(K);std::vector<int>stamp((int64_t)ns*V,-1);
  std::vector<float>cache((int64_t)ns*V*C);
  for(int ray=lo;ray<hi;ray++){
   norm.eval(nu[ray],1.,Y.data());std::vector<double>sum(V,0.);
   for(int64_t n=off[ray];n<off[ray+1];n++){
    const float*q=p+6*n;int ir=base[n]/nc,j=base[n]%nc;
    double ah=q[0],t=q[1],cp=q[2],cm[65];cm[0]=1;
    if(M>1)cm[1]=cp;for(int m=2;m<M;m++)cm[m]=2*cp*cm[m-1]-cm[m-2];
    for(int v=0;v<V;v++)if(mode[v]>=0){
     int count=mode[v]==2?4:2;int start=mode[v]==2?left[j]:j;
     double wr[4];if(count==2){wr[0]=1-t;wr[1]=t;}
     else for(int c=0;c<4;c++){const double*a=poly+16*j+4*c;wr[c]=((a[3]*t+a[2])*t+a[1])*t+a[0];}
     double val=0,logs=0;bool positive=true;
     for(int h=0;h<2;h++)for(int c=0;c<count;c++){
      int cell=(ir+h)*nc+start+c,tag=cell*V+v;
      float*z=cache.data()+(int64_t)tag*C;
      if(stamp[tag]!=ray){
       std::fill(z,z+C,0.f);const float*s=src+(int64_t)cell*S*V+v;
       for(int l=0;l<=2;l++)for(int m=0;m<=l;m++)z[M+m]+=(float)Y[ix(l,m)]*s[ix(l,m)*V];
       for(int l=0;l<=L;l++)for(int m=0;m<=l;m++)z[m]+=(float)Y[ix(l,m)]*s[(6+ix(l,m))*V];
       stamp[tag]=ray;
      }
      double r=0,a=0;for(int m=0;m<M;m++){a+=cm[m]*z[m];if(m<=2)r+=cm[m]*z[M+m];}
      double w=(h?ah:1-ah)*wr[c];
      double value=mode[v]==1?hr[ir+h]*r+ha[ir+h]*a:q[4]*r+q[5]*a;
      val+=w*value;
      if(mode[v]==1){if(value<=0)positive=false;else logs+=w*log(value);}
     }
     if(mode[v]==1&&positive)val=exp(logs);
     sum[v]+=q[3]*val;
    }
   }
   for(int v=0;v<V;v++)out[(int64_t)ray*V+v]=sum[v];
  }
 };
 std::vector<std::thread>ts;for(int k=0;k<RIG_THREADS;k++)ts.emplace_back(worker,nr*k/RIG_THREADS,nr*(k+1)/RIG_THREADS);for(auto&t:ts)t.join();
}
// Stage B preparation only. The Python launcher must first enforce A2_pass.
// Appended to the existing, read-only SH kernel text at build time.
static void tensor_coeff(const double*B,const double*P,double chi,double cx,double sx,double*T){
 double s=sin(chi),c=cos(chi),Q[9]={-c*cx,sx,s*cx,-c*sx,-cx,s*sx,s,0,c},R[9]={},K[9]={};
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)R[3*i+j]+=B[3*k+i]*Q[3*k+j];
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)K[3*i+j]+=P[3*i+k]*R[3*k+j];
 int ii[4]={0,1,2,0},jj[4]={0,1,2,2};
 for(int u=0;u<4;u++){int i=ii[u],j=jj[u];T[4*u]=K[3*i]*K[3*j];T[4*u+1]=K[3*i+1]*K[3*j+1];T[4*u+2]=K[3*i+2]*K[3*j+2];T[4*u+3]=K[3*i]*K[3*j+2]+K[3*i+2]*K[3*j];}
}
// Two-layer tropospheric aerosol (scripts/sosfuse2/tropo_profile.py): selected by aerosol_height<0 only; then
// P=[bl_top-w/2,1/w,a_bl,k_bl,a_ft,k_ft,bl_top] follows asc[0..W) (smoothstep blend over w, same expression as kernel.cu).
// Any other aerosol_height keeps the exponential expression unchanged.
static double aerosol_two_layer(double h,const double*P){
 double x=std::clamp((h-P[0])*P[1],0.,1.),s=x*x*(3.-2.*x);
 return (P[2]*exp(-h*P[3]))*(1.-s)+(P[4]*exp(-(std::max(h,P[0])-P[6])*P[5]))*s;}
extern "C" void full_transport(int nr,int W,int nc,int ns,int mode,int first,int compact,
 const int*active,const int64_t*off,const int32_t*base,const uint8_t*lit,const float*p,
 const double*nu,const double*target,const double*dl,const float*src,const double*ext,
 const double*asc,const double*chi,const double*hr,const double*ha,const double*poly,
 const int*left,const int64_t*goff,const float*gdir,const float*gval,const float*gm,
 double delta,double aerosol_height,double aerosol_g,double*out){
 auto worker=[&](int lo,int hi){
 const int L=32,K=561,S=571,M=33,V=2*W,C=36;Norm norm(L);std::vector<double>Y(K);
  bool has_aerosol=false;for(int w=0;w<W;w++)has_aerosol|=asc[w]!=0;
  std::vector<int>stamp(ns,-1);std::vector<float>cache((int64_t)ns*V*C);
  const double mix=(1-delta)/(1+delta/2),RAY=3/(8*PI);
  for(int ray=lo;ray<hi;ray++){
   const double*B=target+9*ray;const double*d=dl+3*ray;double P[9],ps[9],dp[9]={},direct[4],pu[4];int ii[4]={0,1,2,0},jj[4]={0,1,2,2};
   for(int i=0;i<3;i++)for(int j=0;j<3;j++){P[3*i+j]=(i==j)-d[i]*d[j];ps[3*i+j]=(i==j)-B[6+i]*B[6+j];}
   for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)for(int l=0;l<3;l++)dp[3*i+j]+=P[3*i+k]*ps[3*k+l]*P[3*l+j]/2;
   for(int u=0;u<4;u++){direct[u]=dp[3*ii[u]+jj[u]];pu[u]=P[3*ii[u]+jj[u]];}
   norm.eval(nu[ray],1,Y.data());std::vector<double>sum((int64_t)V*4,0.);double sun_phase=hg(nu[ray],aerosol_g);
   for(int64_t n=off[ray];n<off[ray+1];n++){
    float expanded[18]={};const float*q;
    if(compact){const float*x=p+6*n;expanded[0]=x[0];expanded[1]=x[1];expanded[4]=x[2];expanded[5]=x[3];expanded[16]=x[4];expanded[17]=x[5];q=expanded;}else q=p+18*n;
    int ir=base[n]/nc,j=base[n]%nc;double h=q[5],rh=exp(-std::max(0.,h)/8),ahh=aerosol_height<0?aerosol_two_layer(std::max(0.,h),asc+W):exp(-std::max(0.,h)/aerosol_height);
    double angle=chi[j]+q[1]*(chi[j+1]-chi[j]),T[16];tensor_coeff(B,P,angle,q[16],q[17],T);
    double st=sqrt(std::max(0.,1-nu[ray]*nu[ray])),wx=B[0]*d[0]+B[1]*d[1]+B[2]*d[2],wy=B[3]*d[0]+B[4]*d[1]+B[5]*d[2];
    double cp=st>1e-12?std::clamp((wx*q[16]+wy*q[17])/st,-1.,1.):1.,cm[33];cm[0]=1;cm[1]=cp;for(int m=2;m<33;m++)cm[m]=2*cp*cm[m-1]-cm[m-2];
    int count=(!first&&mode==2)?4:2,start=count==4?left[j]:j;double wr[4];
    if(count==2){wr[0]=1-q[1];wr[1]=q[1];}else for(int c=0;c<4;c++){const double*a=poly+16*j+4*c;wr[c]=((a[3]*q[1]+a[2])*q[1]+a[1])*q[1]+a[0];}
    std::vector<double>ground_aerosol(W,0.);
    if(first){
     double ss=sqrt(std::max(0.,1.-q[2]*q[2])),sp=sqrt(std::max(0.,1.-q[3]*q[3]));
     for(int rr=0;rr<2;rr++)for(int c=0;c<2;c++){
      int cell=(ir+rr)*nc+j+c;double sw=(rr?q[0]:1-q[0])*(c?q[1]:1-q[1]);
      for(int64_t g=goff[cell];g<goff[cell+1];g++){
       double m0=q[2]*gdir[3*g+2]+ss*q[3]*gdir[3*g],m1=ss*sp*gdir[3*g+1];
       double phase=sw*.5*(hg(std::clamp(m0+m1,-1.,1.),aerosol_g)+hg(std::clamp(m0-m1,-1.,1.),aerosol_g));
       for(int w=0;w<W;w++)ground_aerosol[w]+=gval[g*W+w]*phase;
      }
     }
    }
    for(int w=0;w<W;w++){
     double od=0.,sun_od=0.;for(int k=0;k<5;k++){od+=q[6+k]*ext[k*W+w];sun_od+=q[11+k]*ext[k*W+w];}
     double att=q[4]*exp(-od);
     double sun=first&&lit[n]?exp(-sun_od):0;
     for(int lane=0;lane<2;lane++){
     int v=2*w+lane;if(!active[v]&&!first)continue;double val[4]={},logs=0;bool positive=true;double mixv=lane==0?mix:1.;
     if(first){
      double beam_a=ahh*asc[w]*(sun*sun_phase+ground_aerosol[w]);
      for(int u=0;u<4;u++)val[u]=sun*rh*ext[w]*(mixv*RAY*direct[u]+(1-mixv)/(8*PI)*pu[u])+beam_a*pu[u]/2;
     }
     for(int rr=0;rr<2;rr++)for(int c=0;c<count;c++){
      int cell=(ir+rr)*nc+start+c;double sw=(rr?q[0]:1-q[0])*wr[c],tensor[4]={},aero=0,rm[4]={};
      if(first){
       for(int k=0;k<4;k++)rm[k]=gm[((int64_t)cell*4+k)*W+w];
      }else{
       float*z=cache.data()+(int64_t)cell*C*V;
       if(stamp[cell]!=ray){
        std::fill(z,z+C*V,0.f);const float*f=src+(int64_t)cell*S*V;
        for(int l=0;l<=2;l++)for(int m=0;m<=l;m++){float pref=(float)Y[ix(l,m)];for(int v2=0;v2<V;v2++)z[(33+m)*V+v2]+=pref*f[(4+ix(l,m))*V+v2];}
        if(has_aerosol)for(int l=0;l<=32;l++)for(int m=0;m<=l;m++){float pref=(float)Y[ix(l,m)];for(int v2=0;v2<V;v2++)z[m*V+v2]+=pref*f[(10+ix(l,m))*V+v2];}stamp[cell]=ray;
       }
       for(int m=0;m<33;m++)aero+=cm[m]*z[m*V+v];aero*=mode==1?ha[ir+rr]:ahh;
       if(lane==0){for(int k=0;k<4;k++)rm[k]=src[((int64_t)cell*S+k)*V+v];}
       else{double ray=0;for(int m=0;m<=2;m++)ray+=cm[m]*z[(33+m)*V+v];ray*=mode==1?hr[ir+rr]:rh;for(int u=0;u<4;u++)tensor[u]=ray*pu[u]/2;}
      }
      if(first||lane==0){
       double density=(!first&&mode==1)?hr[ir+rr]:rh,tr=rm[0]+rm[1]+rm[2];
       for(int u=0;u<4;u++){double value=0;for(int k=0;k<4;k++)value+=T[4*u+k]*rm[k];tensor[u]=density*(mixv*value+(1-mixv)/3*tr*pu[u]);}
      }
      for(int u=0;u<4;u++){tensor[u]+=aero*pu[u]/2;val[u]+=sw*tensor[u];}
      if(!first&&mode==1){double tr=tensor[0]+tensor[1]+tensor[2];if(tr<=0)positive=false;else logs+=sw*log(tr);}
     }
     if(!first&&mode==1&&positive){double tr=val[0]+val[1]+val[2];if(tr>0){double scale=exp(logs)/tr;for(int u=0;u<4;u++)val[u]*=scale;}}
     if(lane==1){double tr=val[0]+val[1]+val[2];for(int u=0;u<4;u++)val[u]=tr*pu[u]/2;}
     for(int u=0;u<4;u++)sum[4*v+u]+=att*val[u];
     }
    }
   }
   std::copy(sum.begin(),sum.end(),out+(int64_t)ray*V*4);
  }
 };
 std::vector<std::thread>ts;for(int k=0;k<RIG_THREADS;k++)ts.emplace_back(worker,nr*k/RIG_THREADS,nr*(k+1)/RIG_THREADS);for(auto&t:ts)t.join();
}
