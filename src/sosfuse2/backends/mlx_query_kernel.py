"""Metal transport of the four per-ray observer tensor components.

Each multiple-scattering node records a 4x4 rotation (0:16), four spatial
weights (16:20), the base cell bits (20), two Rayleigh and two aerosol
densities (21:25), two mixing coefficients (25:27), the azimuth cosine
series (27:28+L), then the attenuated path weights for all wavelengths.
"""

HELPERS = r"""
inline float beta(Density v, float4 ext, float e4) {
    float4 p = v.p * ext;
    return (((p.x+p.y)+p.z)+p.w)+v.o4*e4;
}
inline void accumulate(float4 block, thread float4& total, thread float4& correction) {
    float4 y=ftz(block-correction);
    float4 updated=ftz(total+y);
    correction=ftz(ftz(updated-total)-y);
    total=updated;
}
inline float sum4(float4 x) { return ((x.x+x.y)+x.z)+x.w; }
template<int nc, int w, int stride, typename F>
inline float4 corners(F p) {
    return ftz(float4(p[0],p[stride*w],p[nc*stride*w],p[(nc+1)*stride*w]));
}
"""

PREPARE = r"""
int w=thread_position_in_grid.x, r=thread_position_in_grid.y;
auto ray=rays+16*r;
float3 d=load3(ray), dl=load3(ray+3);
float4 pu(1.f-dl*dl,-dl.x*dl.z);
float mix=phys[4], iso=(1.f-mix)/3.f;
float4 ext4(ext[w],ext[W+w],ext[2*W+w],ext[3*W+w]);
float e4=ext[4*W+w];
float e5=STRATO ? ext[5*W+w] : 0.f;
float4 direct(0.f);
float phase=0.f;
float sphase=0.f;
float aer_g=phys[5], g2=aer_g*aer_g, hg_num=(1.f-g2)*0.07957747154594767f;
if (FIRST) {
    float3 u(ray[8],ray[11],ray[14]);
    float3 e=u-dl*scalar_dot3(u,dl);
    direct=(mix*0.1193662073189215f)*((pu-float4(e*e,e.x*e.z))*0.5f)
           +((1.f-mix)*0.039788735772973836f)*pu;
    phase=hg_phase(d.z,aer_g,g2,hg_num);
    if (STRATO) sphase=strato_mie_phase(d.z,w,st_phase,W);
}
float4 total(0.f), correction(0.f), block(0.f);
float tau=0.f, tc=0.f, partial=0.f;
if (CONT) {
    auto v=prior+(r*W+w)*11;
    total=load4(v); correction=load4(v+4);
    tau=v[8]; tc=v[9]; partial=v[10];
}
int s0=offsets[r], nodes=(offsets[r+1]-s0)*NQ;
int begin=window[0], end=min(window[1],nodes);
constexpr int RECORD=27+L+1+W;
for (int node=begin;node<end;++node) {
    int seg=node/NQ, k=node%NQ;
    auto rec=segments+(EXTENDED ? 14 : 11)*(s0+seg);
    float3 base=load3(rec);
    float h0=rec[3], pd=rec[4], span=rec[5], top=phys[0];
    float shell=EXTENDED ? rec[10] : 0.f;
    float u=quad[2*k], old_u=k>0?quad[2*(k-1)]:0.f;
    float v=span>0.f?u:1.f-u, prev=span>0.f?old_u:1.f-old_u;
    float a=span*prev*prev, b=span*v*v;
    float h=path_altitude<EXTENDED>(b,pd,h0,shell,top);
    float ha=(EXTENDED && k==0) ? endpoint_support_height(rec[11],rec[12],phys)
                                : path_altitude<EXTENDED>(a,pd,h0,shell,top);
    float hm=path_altitude<EXTENDED>(0.5f*(a+b),pd,h0,shell,top);
    Density cur=density<MASK,LAYER,NT,EXTENDED>(h,tab,phys,layer);
    Density old=density<MASK,LAYER,NT,EXTENDED>(ha,tab,phys,layer);
    Density mid=density<MASK,LAYER,NT,EXTENDED>(hm,tab,phys,layer);
    float st_cur=STRATO ? strato_density<EXTENDED>(h,phys) : 0.f;
    float scale=(abs(span)*(u-old_u))*(prev+v)/6.f;
    float bc=beta(cur,ext4,e4)+(STRATO ? st_cur*e5 : 0.f);
    float bo=beta(old,ext4,e4)+(STRATO ? strato_density<EXTENDED>(ha,phys)*e5 : 0.f);
    float bm=beta(mid,ext4,e4)+(STRATO ? strato_density<EXTENDED>(hm,phys)*e5 : 0.f);
    partial+=scale*((bo+bc)+4.f*bm);
    float depth=(tau-tc)+partial;
    float att=ftz(((2.f*(abs(span)*v))*quad[2*k+1])*ftz(precise::exp(-depth)));
    if (k==NQ-1) {
        float last=span>0.f?span:0.f;
        float hl=EXTENDED ? endpoint_support_height(rec[13],rec[12],phys) : altitude(last,pd,h0,top);
        float hlm=path_altitude<EXTENDED>(0.5f*(last+b),pd,h0,shell,top);
        old=density<MASK,LAYER,NT,EXTENDED>(hl,tab,phys,layer);
        mid=density<MASK,LAYER,NT,EXTENDED>(hlm,tab,phys,layer);
        scale=(abs(span)*(1.f-u))*(v+(span>0.f?1.f:0.f))/6.f;
        float bl=beta(old,ext4,e4)+(STRATO ? strato_density<EXTENDED>(hl,phys)*e5 : 0.f);
        float blm=beta(mid,ext4,e4)+(STRATO ? strato_density<EXTENDED>(hlm,phys)*e5 : 0.f);
        partial+=scale*((bc+bl)+4.f*blm);
        float y=partial-tc, next=tau+y;
        tc=(next-tau)-y; tau=next; partial=0.f;
    }
    float3 pos=fma(float3(b),d,base);
    if (FIRST) {
        bool lit=!(k>=int(rec[8]) && k<int(rec[9]));
        if (att!=0.f && (lit || GROUND)) {
            float sun=0.f;
            if (lit) {
                float td=sqrt(max(0.f,h*(12742.f+h)));
                float numer=pos.z+td;
                if (pos.z<0.f && td>0.f) {
                    // (td+z)(td-z) = rho^2 - R^2, evaluated at the interval anchor.
                    float cyl=fma(b,fma(b,dot(d.xy,d.xy),2.f*rec[7]),rec[6]);
                    numer=cyl/(td-pos.z);
                }
                float sx=sqrt(clamp(numer/(6371.f+h+td),0.f,1.f));
                int i=bracket(sun_h,NSH,h);
                float ah=(h-sun_h[i])/(sun_h[i+1]-sun_h[i]);
                float fx=sx*float(NSX-1);
                int j=clamp(int(fx),0,NSX-2);
                float ax=fx-float(j);
                float4 weights((1.f-ah)*(1.f-ax),ah*(1.f-ax),(1.f-ah)*ax,ah*ax);
                constexpr int NSC=STRATO ? 6 : 5;
                auto sc=sun_col+NSC*(i*NSX+j);
                float opt=0.f;
                for (int z=0;z<5;++z) {
                    float col=sum4(weights*float4(sc[z],sc[NSC*NSX+z],sc[NSC+z],sc[NSC*(NSX+1)+z]));
                    opt+=col*ext[z*W+w];
                }
                if (STRATO) {
                    float st_col=sum4(weights*float4(sc[5],sc[NSC*NSX+5],sc[NSC+5],sc[NSC*(NSX+1)+5]));
                    opt+=st_col*e5;
                }
                sun=ftz(precise::exp(-opt));
            }
            float ga=0.f, sga=0.f;
            float4 val(0.f);
            if (GROUND) {
                float rho=length(pos.xy);
                float ang=precise::atan2(rho,pos.z);
                int gi=bracket(hg,NH,h), gj=bracket(cg,NC,ang);
                float ah=clamp((h-hg[gi])/(hg[gi+1]-hg[gi]),0.f,1.f);
                float ac=clamp((ang-cg[gj])/(cg[gj+1]-cg[gj]),0.f,1.f);
                int base_cell=gi*NC+gj;
                float4 sw=float4(1.f-ah,1.f-ah,ah,ah)*float4(1.f-ac,ac,1.f-ac,ac);
                float2 axis=azimuth(pos.xy,rho);
                ang=cg[gj]+ac*(cg[gj+1]-cg[gj]);
                float sn=precise::sin(ang), cs=precise::cos(ang);
                float3 q0(-cs*axis.x,-cs*axis.y,sn), q1(axis.y,-axis.x,0.f), q2(sn*axis.x,sn*axis.y,cs);
                if (cur.p.y!=0.f || (STRATO && st_cur!=0.f)) {
                    float mu_g=scalar_dot3(d,q2);
                    float exd_g=scalar_dot3(d,q0);
                    float aey_g=abs(scalar_dot3(d,q1));
                    if (HG_TABLE) {
                        int tk, pk;
                        float tw[6], pw[6];
                        float ta=precise::atan2(length(float2(exd_g,aey_g)),mu_g)*(float(HG_BINS)/3.14159265358979323846f);
                        float pa=(aey_g==0.f && exd_g==0.f)?0.f:precise::atan2(aey_g,exd_g)*(float(HG_BINS)/3.14159265358979323846f);
                        hg_stencil(ta,tk,tw);
                        hg_stencil(pa,pk,pw);
                        for (int c=0;c<4;++c) {
                            int cc=base_cell+(c/2)*NC+(c%2);
                            int tidx=g_idx[cc];
                            if (tidx>=0) {
                                size_t off=((size_t(tidx)*HG_NODES+tk)*HG_NODES+pk)*W+w;
                                if (cur.p.y!=0.f) {
                                    auto row=g_f0+off;
                                    float gv_sum=0.f;
                                    for (int jj=0;jj<6;++jj) {
                                        float az_sum=0.f;
                                        for (int ii=0;ii<6;++ii)
                                            az_sum=fma(pw[ii],row[(jj*HG_NODES+ii)*W],az_sum);
                                        gv_sum=fma(tw[jj],az_sum,gv_sum);
                                    }
                                    ga=fma(sw[c],gv_sum,ga);
                                }
                                if (STRATO && st_cur!=0.f) {
                                    auto srow=g_f1+off;
                                    float sgv_sum=0.f;
                                    for (int jj=0;jj<6;++jj) {
                                        float az_sum=0.f;
                                        for (int ii=0;ii<6;++ii)
                                            az_sum=fma(pw[ii],srow[(jj*HG_NODES+ii)*W],az_sum);
                                        sgv_sum=fma(tw[jj],az_sum,sgv_sum);
                                    }
                                    sga=fma(sw[c],sgv_sum,sga);
                                }
                            }
                        }
                    } else {
                        for (int c=0;c<4;++c) {
                            int cc=base_cell+(c/2)*NC+(c%2);
                            int j0=g_idx[cc], j1=g_idx[cc+1];
                            float swc=sw[c];
                            for (int jj=j0;jj<j1;++jj) {
                                auto gd=g_f0+3*jj;
                                float m0=mu_g*gd[2]+exd_g*gd[0];
                                float m1=aey_g*gd[1];
                                float u_p=clamp(m0+m1,-1.f,1.f);
                                float u_m=clamp(m0-m1,-1.f,1.f);
                                float gvw=g_f1[jj*W+w];
                                if (cur.p.y!=0.f) {
                                    float ph=0.5f*(hg_phase(u_p,aer_g,g2,hg_num)+hg_phase(u_m,aer_g,g2,hg_num));
                                    ga=ftz(ga+ftz(ftz(swc*ph)*gvw));
                                }
                                if (STRATO && st_cur!=0.f) {
                                    float sph=0.5f*(strato_mie_lut(u_p,w,st_phase+33*W,W)+strato_mie_lut(u_m,w,st_phase+33*W,W));
                                    sga=ftz(sga+ftz(ftz(swc*sph)*gvw));
                                }
                            }
                        }
                    }
                }
                float ray_s=ftz(ftz(sun*cur.p.x)*ext4.x);
                float beam=ftz(ftz(cur.p.y*asca[w])*ftz(ftz(sun*phase)+ga));
                val=ftz(ftz(ray_s*direct)+ftz(ftz(beam*pu)*0.5f));
                if (STRATO) {
                    float sbeam=ftz(ftz(st_cur*asca[W+w])*ftz(ftz(sun*sphase)+sga));
                    val=ftz(val+ftz(ftz(sbeam*pu)*0.5f));
                }
                if (cur.p.x!=0.f) {
                    float3 x0=reference_row(load3(ray+6),dl.x,d,q0,q1,q2);
                    float3 x1=reference_row(load3(ray+9),dl.y,d,q0,q1,q2);
                    float3 x2=reference_row(load3(ray+12),dl.z,d,q0,q1,q2);
                    float4 t0=rotation(x0,x0), t1=rotation(x1,x1), t2=rotation(x2,x2), t3=rotation(x0,x2);
                    float4 rm(0.f);
                    for (int c=0;c<4;++c) {
                        int cc=base_cell+(c/2)*NC+(c%2);
                        auto gp=gm+(cc*4)*W+w;
                        rm+=sw[c]*float4(gp[0],gp[W],gp[2*W],gp[3*W]);
                    }
                    float tr=(rm.x+rm.y)+rm.z;
                    float iso_tr=ftz(iso*tr);
                    float4 rot(dot(t0,rm),dot(t1,rm),dot(t2,rm),dot(t3,rm));
                    val=ftz(val+ftz(cur.p.x*ftz(ftz(mix*rot)+ftz(iso_tr*pu))));
                }
            } else {
                float ray_s=ftz(ftz(sun*cur.p.x)*ext4.x);
                float beam=ftz(ftz(cur.p.y*asca[w])*ftz(sun*phase));
                val=ftz(ftz(ray_s*direct)+ftz(ftz(beam*pu)*0.5f));
                if (STRATO) {
                    float sbeam=ftz(ftz(st_cur*asca[W+w])*ftz(sun*sphase));
                    val=ftz(val+ftz(ftz(sbeam*pu)*0.5f));
                }
            }
            block=ftz(block+ftz(val*att));
        }
        if (k==NQ-1 || node+1==end) {
            accumulate(block,total,correction);
            block=float4(0.f);
        }
    } else {
        auto pt=factors+size_t(s0*NQ+node-begin)*RECORD;
        pt[27+L+1+w]=att;
        if (w) continue;
        float rho=length(pos.xy);
        float ang=precise::atan2(rho,pos.z);
        int i=bracket(hg,NH,h), j=bracket(cg,NC,ang);
        float ah=clamp((h-hg[i])/(hg[i+1]-hg[i]),0.f,1.f);
        float ac=clamp((ang-cg[j])/(cg[j+1]-cg[j]),0.f,1.f);
        float2 axis=azimuth(pos.xy,rho);
        float cp=ray[15]>1e-12f?clamp((d.x*axis.x+d.y*axis.y)/ray[15],-1.f,1.f):1.f;
        ang=cg[j]+ac*(cg[j+1]-cg[j]);
        float sn=precise::sin(ang), cs=precise::cos(ang);
        float3 q0(-cs*axis.x,-cs*axis.y,sn), q1(axis.y,-axis.x,0.f), q2(sn*axis.x,sn*axis.y,cs);
        float3 x0=reference_row(load3(ray+6),dl.x,d,q0,q1,q2);
        float3 x1=reference_row(load3(ray+9),dl.y,d,q0,q1,q2);
        float3 x2=reference_row(load3(ray+12),dl.z,d,q0,q1,q2);
        store4(pt,rotation(x0,x0)); store4(pt+4,rotation(x1,x1));
        store4(pt+8,rotation(x2,x2)); store4(pt+12,rotation(x0,x2));
        store4(pt+16,float4(1.f-ah,1.f-ah,ah,ah)*float4(1.f-ac,ac,1.f-ac,ac));
        bool out_air=EXTENDED && (h>phys[10]);
        bool out_low=EXTENDED && (h>phys[9]);
        int cell_bits=i*NC+j;
        if (EXTENDED && out_low) cell_bits|=0x40000000;
        pt[20]=as_type<float>(cell_bits);
        pt[21]=out_air ? 0.f : dens[2*i]; pt[22]=out_air ? 0.f : dens[2*i+2];
        pt[23]=out_low ? 0.f : dens[2*i+1]; pt[24]=out_low ? 0.f : dens[2*i+3];
        pt[25]=mix; pt[26]=iso;
        pt[27]=1.f; pt[28]=cp;
        for (int m=2;m<=L;++m) pt[27+m]=(2.f*cp)*pt[26+m]-pt[25+m];
    }
}
if (FIRST || SAVE) {
    auto v=out+(r*W+w)*(SAVE?11:4);
    store4(v,total);
    if (SAVE) { store4(v+4,correction); v[8]=tau; v[9]=tc; v[10]=partial; }
}
"""


MODES = r"""
constexpr int K=(L+1)*(L+2)/2;
constexpr int STRIDE=10+(STRATO ? 2*K : K);
int w=thread_position_in_grid.x, m=thread_position_in_grid.y, pair=thread_position_in_grid.z;
if (w>=W || m>L) return;
int r=pairs[2*pair], c=pairs[2*pair+1];
auto src=field+(size_t(c)*STRIDE+10)*W+w;
float v=0.f;
if (COMBINE) {
    int ir=c/NC;
    float hr_v=st_comb[3*ir];
    float ha_c=st_comb[3*ir+1];
    float hs_c=st_comb[3*ir+2];
    if (SKIP_THIN) {
        float cut=1e-7f*hr_v;
        ha_c=ha_c>cut ? ha_c : 0.f;
        hs_c=hs_c>cut ? hs_c : 0.f;
    }
    if (ha_c!=0.f || hs_c!=0.f) {
        int k=m*(m+1)/2+m;
        for (int l=m;l<=L;++l) {
            float cv=(ha_c!=0.f ? src[k*W]*ha_c : 0.f)+(hs_c!=0.f ? src[(K+k)*W]*hs_c : 0.f);
            v=ftz(v+ftz(basis[r*K+k]*ftz(cv)));
            k+=l+1;
        }
    }
} else {
    int k=m*(m+1)/2+m;
    auto srcp=src+(PLANE*K)*W;
    for (int l=m;l<=L;++l) {
        v=ftz(v+ftz(basis[r*K+k]*ftz(srcp[k*W])));
        k+=l+1;
    }
}
out[(size_t(pair)*(L+1)+m)*W+w]=v;
"""


MULTIPLE = r"""
constexpr int K=(L+1)*(L+2)/2, STRIDE=10+(STRATO ? 2*K : K), RECORD=28+L+W;
int x=thread_position_in_grid.x, q=thread_position_in_grid.y, r=thread_position_in_grid.z;
int lane=x%WN, iw=x, w=min(iw,W-1);
float3 dl=load3(rays+16*r+3);
float4 pu(1.f-dl*dl,-dl.x*dl.z), total(0.f), correction(0.f);
if (CONT) {
    auto v=prior+(r*W+w)*8;
    total=load4(v); correction=load4(v+4);
}
int s0=offsets[r], nodes=(offsets[r+1]-s0)*NQ;
int begin=window[0], end=min(window[1],nodes);
int previous=-1;
float4 rm0,rm1,rm2,rm3,hr,ha,hs,hs_cell,isotropic;
threadgroup float4 contributions[PN*WN];
for (int chunk=begin;chunk<end;chunk+=PN) {
    int node=min(chunk+q,end-1);
    auto pt=factors+size_t(s0*NQ+node-begin)*RECORD;
    float att=pt[27+L+1+w];
    int raw_cell=as_type<int>(pt[20]);
    bool out_low=EXTENDED && ((raw_cell & 0x40000000)!=0);
    int cell=EXTENDED ? (raw_cell & 0x1FFFFFFF) : raw_cell;
    float mix=pt[25], iso=pt[26];
    auto src=field+size_t(cell)*STRIDE*W+w;
    if (cell!=previous) {
        rm0=corners<NC,W,STRIDE>(src);
        rm1=corners<NC,W,STRIDE>(src+W);
        rm2=corners<NC,W,STRIDE>(src+2*W);
        rm3=corners<NC,W,STRIDE>(src+3*W);
        if (STRATO_SEP) {
            int ir=cell/NC;
            hs_cell=float4(st_comb[3*ir+2],st_comb[3*ir+2],st_comb[3*(ir+1)+2],st_comb[3*(ir+1)+2]);
        }
        isotropic=ftz(iso*ftz(ftz(rm0+rm1)+rm2));
        previous=cell;
    }
    hr=float4(pt[21],pt[21],pt[22],pt[22]);
    ha=float4(pt[23],pt[23],pt[24],pt[24]);
    if (STRATO_SEP) hs=out_low ? float4(0.f) : hs_cell;
    float4 aer(0.f);
    float4 saer(0.f);
    if (!out_low) {
        auto map=slots+4*(s0*NQ+node-begin);
        constexpr int SW=(L+1)*W;
        auto a0=modes+size_t(map[0])*SW+w, a1=modes+size_t(map[1])*SW+w;
        auto a2=modes+size_t(map[2])*SW+w, a3=modes+size_t(map[3])*SW+w;
        float4 acc(0.f);
        #pragma unroll
        for (int m=0;m<=L;++m) {
            int k=m*W;
            float4 v(a0[k],a1[k],a2[k],a3[k]);
            acc=ftz(acc+ftz(pt[27+m]*v));
        }
        aer=ftz(acc*ha);
        if (STRATO_SEP) {
            auto sa0=smodes+size_t(map[0])*SW+w, sa1=smodes+size_t(map[1])*SW+w;
            auto sa2=smodes+size_t(map[2])*SW+w, sa3=smodes+size_t(map[3])*SW+w;
            float4 sacc(0.f);
            #pragma unroll
            for (int m=0;m<=L;++m) {
                int k=m*W;
                float4 v(sa0[k],sa1[k],sa2[k],sa3[k]);
                sacc=ftz(sacc+ftz(pt[27+m]*v));
            }
            saer=ftz(sacc*hs);
        }
    }
    float4 val, weights=load4(pt+16);
    float4 trace(0.f);
    for (int k=0;k<4;++k) {
        float4 a=load4(pt+4*k);
        float4 rr=ftz(ftz(a.x*rm0)+ftz(a.y*rm1));
        rr=ftz(rr+ftz(a.z*rm2));
        rr=ftz(rr+ftz(a.w*rm3));
        float4 t=ftz(hr*ftz(ftz(mix*rr)+ftz(isotropic*pu[k])));
        t=ftz(t+ftz(ftz(aer*pu[k])*0.5f));
        if (STRATO_SEP) t=ftz(t+ftz(ftz(saer*pu[k])*0.5f));
        val[k]=sum4(ftz(weights*t));
        if (k<3) trace=ftz(trace+t);
    }
    bool4 good=trace>=0x1p-126f;
    float4 tr=select(float4(1.f),trace,good), wt=select(float4(0.f),weights,good);
    float sumw=sum4(wt), am=sum4(ftz(wt*tr));
    if (sumw>0.f && am>=0x1p-126f) {
        float logs=sum4(wt*fast::log(tr));
        val=ftz(val*fast::exp(logs-sumw*fast::log(am/sumw)));
    }
    contributions[q*WN+lane]=ftz(val*att);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!q) {
        float4 block(0.f);
        for (int i=0;i<min(PN,end-chunk);++i) block=ftz(block+contributions[i*WN+lane]);
        accumulate(block,total,correction);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (!q && iw<W) {
    auto v=out+(r*W+w)*(SAVE?8:4);
    store4(v,total);
    if (SAVE) store4(v+4,correction);
}
"""
