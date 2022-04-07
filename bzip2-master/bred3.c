/* PROGRAM for COMPRESSION */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <sys/time.h>

/* USE MACRO DEF  "#define long int"  
if int has four or more BYTES  */

#define FL fflush(stdout) ;
#define TR(w,x,y,z) { static long\
 trct=500 ; if (trct-->0)\
 printf("\n x %w y %w z %w",x,y,z) ;\
 if (trct==0) printf("end trace x,y,z") ;\
 FL }

#define VS(v,op,n) {long nZ=0 ; while (nZ<n) \
(v)[nZ] op, nZ++ ; }

#define GPNO 8
#define GPSZ 25
#define DEPTH 32

   /* OMIT DEF IF CIPHERING IS NOT NEEDED */
#define CIPHER k4=key[k6^k4] ;\
               k5=key[k6^k5] ; k3+=k4^k5 ;

#ifdef CIPHER
#define PUT(a,b) { bff=bff<<(b)^(a) ; bfct+=b ; \
 while (bfct>7) { bfct-=8 ; putc(k6=k3^bff>>bfct,fpw) ;\
 CIPHER outct++ ; } }

#else
#define PUT(a,b) { bff=bff<<(b)^(a) ; bfct+=b ; \
 while (bfct>7) { \
 bfct-=8 ; putc(bff>>bfct,fpw) ; outct++ ; } }
#endif

/* a quick sort routine using a given pointer
list starting at s and ending at e. The
pointers scan puts rch character string
in order ( backwards dictionary ). Work
is decreased by the sorting work at a depth
greater than DEPTH, default 64 and returned
as the result. If it becomes -ve the sort 
is stopped and the result is zero. The sorting
will use characters below rch[0] if necessary
and there should be enough space to allow this
without going into uncharted space.  */

long qsw(long *s,long *e,
      unsigned char *ch,long work) {
long zref,zs,ze,*ss,*ee,dd=0 ;
unsigned char *chd=ch,* list[99] ;   
goto jss ;

jred : if (dd==0) return work ;
chd=(unsigned char *) list[--dd] ;
e=(long *) list[--dd] ;
s=(long * ) list[--dd] ;
jss :
if (s+1>=e) {unsigned char *chdd=chd ;
  if (e==s) goto jred ;
  while (chdd[*s]==chdd[*e]) chdd-- ;
  if (chdd[*s]>chdd[*e])
  zref= *s, *s= *e, *e=zref ;
  zref=ch-chdd-DEPTH ;
  if (zref>0) { work-=zref ;
  if (work<0) return 0 ; }
  goto jred ; }

if ((zs=chd[*s])==(ze=chd[*e])) {
  ss=s+1 ;
  while ( (chd[*ss]==ze) && (ss!=e) ) ss++ ;
  if (ss==e)   {
    chd-- ; zref=ch-chd-DEPTH ;
    if (zref>0)               {
       work-=zref ;
       if (work<0) return 0 ; }
    goto jss ; }
  else zs=chd[*ss] ; }
zref=(zs+ze)>>1 ; ss=s ; ee=e ; goto js2 ;

jss1 :
zs= *ss ; *ss++ = *ee ; *ee-- =zs ;
js2 :
while (zref>=chd[*ss]) ss++ ;
while (zref<chd[*ee]) ee-- ;
if (ss<ee) goto jss1 ;

ss-- ; ee++ ;
if (s-ss)
  if (e-ee)
     if (ss-s>e-ee) {
        list[dd++]=(unsigned char *) s ;
        list[dd++]=(unsigned char *) ss ;
        list[dd++]= chd ;
        s=ee ; goto jss ; }
     else {
        list[dd++]=(unsigned char *) ee ;
        list[dd++]=(unsigned char *) e ;
        list[dd++]= chd ;
        e=ss ; goto jss ; }
  else { e=ss ; goto jss ; }
else if (e-ee) { s=ee ; goto jss ; }
      else goto jred ;             }


/* A Sort routine that generates a pointer list
from pt to e, which points to rch in backwards
dictionary order. Work is decreased by the sorting
done at DEPTH or greater by the subroutine qsw
and given as the result. If -ve the result is
forced zero and the sorting aborted. The sorting
result assumes that rch[0] is preceded by a 
character lower in value than all others. */

long qsww( long * pt, long * e, unsigned char * rch,
   long work)
{{long tft[65540],x,y,z,nlim=e-pt+1,
 *ftt,mm,p,*s,m, n, startoffset=0,
 ftm[257], *ft ;
 rch[nlim]=0 ;
for (m=0 ; m<65540 ; m++ ) tft[m]=0 ;
ft=tft+2 ; x=0 ;
for ( m=0 ; m<nlim ; m++ )
  ft[rch[m]<<8^x]++,
  x=rch[m] ;
x=0 ;
for ( m=0 ; m<65536 ; m++ )
x=ft[m]+=x ;
ft-- ;
x=0 ;
for (m=0 ; m<nlim ; m++ )
  pt[ft[rch[m]<<8^x]++]=m,
  x=rch[m] ;
ft-- ;
 while (rch[startoffset]==0) startoffset++ ;
ft[0]=startoffset ;

for (mm=0,m=0 ; m<256 ; mm+=256,m++)
 if (ft[mm]^ft[mm+256])          {
  for ( n=m ; n<256 ; n++) ftm[n]=ft[m^n<<8] ;
  ftt=ft+mm ;
  for (p=m+1 ; p<256 ; p++ )
    if (ftt[p]+1<ftt[p+1]) {
    work=qsw(pt+ftt[p],pt+ftt[p+1]-1,rch-2,work) ;
    if (work<1) return  0 ; }

 if ( ftt[m+1]-ftt[m] < 10+(ftt[256]-ftt[0]>>3) ) 
  if (ftt[m]+1<ftt[m+1]) {
      work=qsw(pt+ftt[m],pt+ftt[m+1]-1,rch-2,work) ;
      if (work<1) return 0 ;
      goto skip ;        }
   { long *t=pt+ftt[m] ; s=pt+ftt[0] ;
    rch[nlim]=255-m ;
    while (s!=t) { if (rch[*s+1]==m)
             *t= *s+1, t++ ; s++ ; }

    s=pt+ftt[256]-1 ; t=pt+ftt[m+1]-1 ;
    while (s!=t) { if (rch[*s+1]==m) 
                  *t= *s+1, t-- ; s-- ; }
    rch[nlim]=0 ;
   }
skip :
 if (m==0)
   ftm[rch[startoffset]]++ ;
 for ( s=pt+ftt[0] ; s<pt+ftt[256] ; s++ ) {
   z= rch[*s+1] ;
   if (z>m)
   pt[ftm[z]++]= *s+1 ;                    }
                                  }
return work ;
}}

   /* SIMPLE SHELL SORT BY POINTER */
shell(long * v,long * pt,long n) {
long p,q,x,k,pk ;
for (k=(n+3)/5 ; k>0 ; k=(k+1)/3)
  for (p=0 ; p<n-k ; p++ )
    if ( (x=v[pt[p+k]]) < v[pt[p]] )  {
      q=p-k ; pk=pt[p+k] ; pt[p+k]=pt[p] ;
      while ( q>=0 && v[pt[q]]>x )
        pt[q+k]=pt[q], q-=k ;
      pt[q+k]=pk ;                    } 
                                    }

 /* MAKES CODE TABLES FROM FREQUENCIES */
long huffgen(long * f,long *len,long n)
{long j,m,mlim=0,u=0 ,
p,q,x,y,z,bits,r=22,
*pt, *type, cts[50] , *sum ;
pt = ( long * ) malloc((n+1)*sizeof(long)) ;

for (j=n-1 ; j>=0 ; j-- )  {
   if (x= f[j] ) pt[mlim++]=j, u+=x ;
   len[j]=0 ;           }
if (u==0) return 0 ;

type = ( long * ) malloc(sizeof(long)*2*mlim) ;
sum=type+mlim ;
shell(f,pt,mlim) ;
js :
for (j=0 ; j<50 ; j++) cts[j]=0 ;
for (m=0 ; m<mlim ; m++) sum[m]=0x7000000 ;
m=0 ; p=0 ; q=0 ;

while (q<mlim-1)
  if 
( (m+1<mlim) && ( (x=f[pt[m+1]])<=sum[p] ) )
     sum[q]=x+f[pt[m]],
     type[q++]=0,
     m+=2 ;
  else if 
( (m<mlim) && (sum[p+1]< f[pt[m]]) || (m==mlim) )
     sum[q]=sum[p]+sum[p+1],
     type[q++]=2,
     p+=2 ;
        else sum[q]=sum[p++]+f[pt[m++]],
             type[q++]=1 ;

bits=0 ; 
for (j=0 ; j<mlim-1 ; j++) bits+=sum[j] ;
sum[q-1]=0 ;
while (q) { q-- ;
 x=type[q] ; y=sum[q]+1 ;
 if (x==2) sum[--p]=y, sum[--p]=y ;
 else if (x==1)
      len[pt[--m]]=y, sum[--p]=y, cts[y]++ ;
      else len[pt[--m]]=y, 
           len[pt[--m]]=y, cts[y]+=2 ;
         }
y=len[pt[0]]-23 ; if (y>0) {
x=u>>(r--) ; x++ ;
z=0 ; q=0 ;
while ( z>=0) { y=f[pt[q]] ; f[pt[q++]]=x ;
  z+=x-y ; 
              }
f[pt[q-1]] -= z ; goto js ; }

x=0 ;
for (m=49 ; m>=0 ; m--)
   y=cts[m], cts[m]=x, 
   x+=y, x>>=1 ;
for (m=n-1 ; m>=0 ; m--)
   if (len[m])
   f[m]=cts[len[m]]++ ;
free( (char *) pt) ;
free( (char *) type) ;
return bits ;
}

/*  MAIN PRORAM */

long main (long argc, char *argv[])
{
char  chl[50], *error ;
unsigned char *rch ;

#ifdef CIPHER
   /*  SALT/KEY */
struct timeval tp ;
struct timezone tzp ;
long salt1, salt2 ;
static unsigned char k1,k2,k3,k4,k5,k6,
  key[256] ;
#endif

static long r[258],
  space[520*GPNO],
  slist[GPNO],
  lenco[GPNO+1],
  x,y,z,
* trel,   /* holds relative frequencies,
                then codes */
* len,    /* holds code lengths */
* pt,
  m,n,p,q,
  argcc=0,
  chsz,
  silent=1,  /* control variables */
  keep=1,
  errct=0, 
  errlim=1,
  blocksize=200000 ,/* default size */
  work=40000000, /* limits sorting */
  gpno=1,  /* used for multiple tables */
  gpszz=0,
  gpsz,
  grep,
  grepp,
  selct,
  outct, /* counts */
  inct,  /*  incount */
  dct ;  /* table counts */
FILE *fpr, *fpw, *fopen() ;

while (++argcc<argc)
{
    /* ARG HANDLING */
x=0 ;
while ( chl[x]=argv[argcc][x] ) x++ ;
if (chl[0]=='-')     {long zn=2 ;
   x=0 ;
   while (chl[zn]) x=10*x + chl[zn++]-'0' ;
   switch ( argv[argcc][1] )       {
case 's' : silent=x ; break ;
case 'k' : keep=x ; break ;
case 'K' : blocksize=1000*x ; break ;
case 'e' : errlim=x ; break ;
case 'w' : work=x ; break ;
         /* safe size limited decoding for -m */
case 'm' : gpno=x/100 &7 ; gpno++ ;     
           gpszz= x%100 ;
           gpszz/=10 ; gpszz&=3 ; grep=x%10 ;
           break ;

#ifdef CIPHER
case 'p' : if ( *(argv[argcc]+2) == 0 ) {
           /* disable key */
                k2=k3=k4=k5=0 ;
                for (x=0 ; x<256 ; x++)
                key[x]=0 ; }
           /* enable key */
           else k2=argcc ;
           break ;
#endif
default : { error="option wrong" ; errct=errlim ;
             goto err ; }
                                       }
continue ;          }

           /* FILE HANDLING */
if ((fpr=fopen(argv[argcc],"r"))==NULL)
   { error= "cannot open to read" ;
   goto err2  ; }
chl[x++]='.' ; chl[x++]=0 ;

if ((fpw=fopen(chl,"w"))== NULL)
   { error="cannot open to write" ;
   goto err1 ; }

        /* SPACE ALLOCATION */
if (blocksize%1000==0)         {
if (blocksize==0) blocksize=64 ;
   pt= ( long *) malloc(blocksize*(1+sizeof( long ))+1032) ;
   if (pt==0)
       { error= "not enough space from malloc" ;
                 goto err ; }
   rch= (unsigned char * ) pt ;
   rch+=blocksize*sizeof(long)+1032 ;
   blocksize-=10 ; /* effective size 13 less */
   VS(rch-1032,=0,1033)      }

     /* CHARACTER HANDLING */
{{{
long  mlim, bsize=blocksize ;
static long bff,bfct ;

restart : /* start again with bsize reduced */

#ifdef CIPHER
k1= (k2) ? 32 : 0 ;
VS(key,=0,256) k3=0 ;
gettimeofday(&tp,&tzp) ;
salt1+= tp.tv_sec ;
salt2+= tp.tv_usec ;
#endif

inct=outct=dct=selct=bfct=0 ;

    /* READ FILE TO rch */
start :         /* next block of file */
VS(r,=0,258)
m=1 ;
while (m<bsize-2)    {
  z=getc(fpr) ; 
  if (z==EOF) break ;
  r[z]=1 ;
  rch[m++]=z ;       }
inct+= m-1 ;
mlim=m ;

/* CHECK for LONG ZERO STRINGS */
{ long * rw,m,p,q ;
rw=( long * ) rch ;
p=0 ; q=1000/sizeof(long) ;
for (m=1 ; m<mlim/sizeof(long) ; m++)
  if ( rw[m]) { if (p>q) q=p ; p=0 ; }
  else p++ ;
if (q>1000/sizeof(long)) {
   VS(rw-q-5,=0,q-240)  /* extend prefix of zeroes */
  if ( blocksize-mlim<q-240) {
    bsize=mlim-q+1000/sizeof(long)-10 ;
    if (silent&2)
      printf(
      "\n long zero strings %d reduce blocksize to %d"
      ,4*q,bsize) ;
  rewind(fpr) ;
  rewind(fpw) ;
  goto restart ;} } }


   /* SORT BY POINTER */
x=qsww(pt,pt+mlim-1,rch,work) ;

if (x==0)
/* sorting too slow, reduce block size */
 {
  bsize=x=mlim+20>>1 ;
  if (silent&2)
  printf(
  "\nSort too slow, halve block size to %d",x) ;
  rewind(fpr) ;
  rewind(fpw) ;
  goto restart ;
  }

{
long last,n,m, sh, ct ;

    /* SEND GROUP NUMBER,SIZE,
    AND IF CYPHERED */
gpsz=25<<gpszz ;
if (mlim>>13 <gpno) gpno=1+(mlim>>13) ;

#ifndef CIPHER
PUT(224^(gpno-1&7)<<2^gpszz,8) ;
#else
PUT(224-k1^(gpno-1&7)<<2^gpszz,8) ;
if (k1) {

   /* USE SALT AND PASSWORD TO MAKE
    KEY TABLE FOR CIPHERING */
  unsigned char temp[256] ;
  char * kch ;

  gettimeofday(&tp,&tzp) ;
  salt2+= tp.tv_sec+mlim ;
  salt1+= tp.tv_usec ;

  for (m=1 ; m<mlim ; m+=16+mlim>>4)
  salt1+=salt2+pt[m],
  salt2+=salt1/3 ;

  for ( m=0 ; m<8 ; )  {
    putc(temp[m++]=salt1&255,fpw) ;
    salt1 = (salt1>>4)+salt2 ;
                        }
  outct+=8 ;
  kch=argv[k2] ; 
  while ( temp[m++]= *kch++ ) ;
  n=m ;
  while (m<256) 
     temp[m]= temp[m-1]>>3^temp[m-1]<<5^temp[m-n], m++ ;

    /* use randomised temp to make perm key
     so that key to password is difficult */
  {  unsigned char a,b,c,d ;
    a=temp[91],b=temp[37], c=temp[161], d=temp[241] | 1 ;
    for (m=0 ; m<256 ; m++) 
      key[m^a]=b^c, b+=d ;

  /* permute key by temp */
  k3=key[0] ; m=0 ; while (1) {
    k4=temp[m]^key[m] ;
    key[m]=key[k4] ;
    m++ ; if (m==256) break ;
    key[k4]=key[m] ; }
  key[k4]=k3 ;

  k4=key[87], k5=key[78], k3=temp[157] ;
  k1=0 ;
   }     }
#endif

/* FIND USED CHARACTERS AND MAKE CONSECUTIVE  */ 
  /* arrange to count table cost */
dct-=outct-1 ;
r[256]=1 ;
for (n=m=sh=ct=0, last=2 ; n<258 ; n++ ) {
  z=r[n] ;
  if (z^last)  {
    while (ct) {
      ct-- ;
      PUT( (z^1)<<1 ^ ct&1,2 ) ;
      ct>>=1 ; }
            }
  ct++ ;
  if (z) r[m++]=n ;
  last=z ;                            }

chsz=m+2 ; /* allow for 1 2 coding, EOF */
}
 
    /* For the 'next' or predicted values
    CONSTRUCT MOVE TO FRONT values and
    1-2 zero string coding. insert coding
    in pt */

{ long
s=0,j ,k=0 ;
trel=space ;
len=trel+chsz ;

VS(trel,=0,chsz) /* clear frequencies */
for (j=0 ; j<mlim ; j++ )  {
   z=rch[(x=pt[j]+1)] ;
   if (x==mlim) z=256 ;
   p=0 ;
   while (z^r[p]) p++ ;
   q=p ;
   while (p)  r[p]=r[p-1], p--  ;
   r[0]=z ;

   if (q==0) s++ ;
   else { while (s)    {
         s-- ;
         trel[s&1]++ ;
         pt[k++]=s&1 ;
         s>>=1 ;       }
       trel[q+1]++ ;
       pt[k++]= q+1 ;
        }                  }
    /* possible residue, complete
      also set 'eof'  */
while (s)             {
       s-- ;
       trel[s&1]++ ;
       pt[k++]=s&1 ;
       s>>=1 ;        }
 
pt[k++]=chsz-1 ;

      /* GENERATE MULTIPLE TABLES */
if (gpno>1)
{{long sm=0, sms[GPNO], old=1<<30,
   n,t,select,mm, q=0,smss[GPNO];

    /* SET len for START SCAN with
    groups aiming to be consecutive and
    of the same size  */
{
long kk=k, m, n, p, quota, sum[GPNO+GPNO+2] ;
sum[0]=0 ; trel[chsz-1]=1 ;
for (m=n=0 ; n<chsz ; )  {
        if (gpno==m/2) { error="length start calcs wrong" ;
        goto err ;     }
  quota= kk/(gpno-m/2) ;
  while (sum[m]<quota) sum[m]+=trel[n++] ;
  kk-=sum[m] ;
  m++ ; sum[m++]=n ; sum[m]=0 ;
                         }
for (p=0, n=0 ; p<m ; p+=2 )
  for ( ; n<sum[p+1] ; n++ )
     kk=14*sum[m-2]/sum[p]+1,
     kk<<=p+p,
     len[n]= ~kk ;
}

    /* clear all frequency counts */ 
for (n=0 ; n<gpno*chsz<<1 ; n+=chsz<<1 )
VS(trel+n,=0,chsz)
grepp=0 ;

repeat :
VS(smss,=0,gpno)
n=0 ; p=1 ; m=0 ; select=0 ;

while (n<k ) {
  long w ;
  for (w=0 ; w<gpno ; w++ )
    sms[w]=smss[w] ;
  n+=gpsz ; 
  if (n>k) n=k ;
  for (mm=m ; m<n ; m++) {
    sm=len[pt[m]] ;
    for (t=0 ; t<gpno ; t++)
      sms[t]+= sm&15, sm>>=4 ;
                         }
  sm=sms[select] ;
  for (t=0 ; t<gpno ; t++) 
    if (sms[t]<sm)
      sm=sms[t],
      select=t ;

  rch[p++]=select ;      
  trel=space+ select*chsz*2 ;
  for ( ; mm<n ; mm++ )
    trel[pt[mm]]++ ;
              }

y=0 ;
trel=space ;
for (m=0 ; m<gpno*chsz<<1 ; m+=chsz+chsz )
  trel[m]++,
  trel[chsz+m-1]=1,
  y+=huffgen(trel+m,len+m,chsz) ; 
if (silent&4) {
long f[GPNO], m ;
for (m=0 ; m<gpno ; m++) f[m]=0 ;
for (m=1 ; m<p ; m++) f[ rch[m] ]++ ;
printf("\nself ") ;
for (m=0 ; m<gpno ; m++) printf(" %4d",f[m]) ;
              }
if (silent&4 && grepp)
 printf("\nIT GAIN %4d millibits old %d y %d\n",
     (old-y)*1000/mlim,old,y) ;

/* end test */
if ( (y+mlim/1000>=old) || (grepp++==grep) )
       /*GENERATE SELECTOR TABLE FOR TABLES
       AND SEND */
{{long 
  freq[GPNO+1],len[GPNO+1],
  pp=p,m,t, q, ct=0 ;
for (m=0 ; m<gpno ; m++)
  slist[m]=m,
  freq[m]=1 ;
freq[gpno]=1 ;

for (m=1 ; m<p ; m++ ) {
  z=rch[m],
  t=0 ; if (silent&8) { if (m%25==1) putchar('\n') ;
                      printf(" %c",'0'+z) ; }
while (z^slist[t]) t++ ;
q=t ;
while (t) slist[t]=slist[t-1], t-- ;
slist[0]=z ;
if (q==0) ct++ ;
if ( (q>0) || (m==p-1) )
  while (ct) ct--, freq[ct&1]++,
    rch[pp++]=ct&1, ct>>=1 ;
if (q>0) freq[q+1]++,
  rch[pp++]=q+1, rch[pp++]=z ;
                       }
selct+=huffgen(freq,len,gpno+1) ;
for (m=0 ; m<gpno+1 ; m++ )  {
  lenco[m]=q=len[m]-1<<3 ^ freq[m] ;
  PUT(q,6)                }
goto jn ;
}}

old=y ;
   /* SET NEW LENGTHS FOR SEARCH, CLEAR trel */
for (m=0 ; m<chsz ; m++)
   for (n=0 ; n<chsz*gpno<<1 ; n+=chsz<<1 ) {
     long zn ;
     z=len[n+m] ;
     if (z==0) z=len[chsz-1+n] ;
     if (n==0) zn=z ;
     z+=8-zn ; z&=15 ;  /* can overflow rare? */
     len[m]=len[m]<<4^z ;
     trel[n+m]=0 ;                          }

 goto repeat ;
}}

/* SINGLE TABLE */
else
{{
trel[0]++ ;
trel[chsz-1]=1 ; /* make eof smallest code */
y=huffgen(trel,len,chsz) ;
}}

    /* SEND LENGTH TABLES FOR DECODER */
jn :

/* output the code length tables
 3 gives extension
 2 gives +
 1 gives - 
 0 gives equals */ 

{ long last=2 ;
for (m=0 ; m<chsz ; m++ ) {
  z=len[m] ;
  if (z==0) z=len[chsz-1]+1 ;
  while (z>last+1 )
     { last++ ; PUT(3,2)  }
  while (z<last-1 )
     { last-- ; PUT(3,2)  }  
  if (z==last+1) {
     last++ ;
     PUT(2,2) 
     continue ;  }
  if (z==last-1) {
     last-- ;
     PUT(1,2) 
     continue ;  }
  PUT(0,2)               }
}

len+=chsz+chsz ;
if (len < space+gpno*2*chsz)  goto jn ;
len=space+chsz ;
dct+=outct ;

    /* PUT CODED VALUES */
/* MULTIPLE tables */
if (gpno>1) { 
 long n=0, ss=0 ;
 m=0 ; /* p is start of selectors */
 while (n<k) {
  z=rch[p++] ;
    PUT(lenco[z]&7,(lenco[z]>>3)+1) 
  if (z<2)
    n+= gpsz<<ss++ +z ;
  else { 
  z=rch[p++] ;
  trel=space+2*z*chsz ;
  len=trel+chsz ;
  n+=gpsz ;
  ss=0 ;
       }
  if (n>=k) n=k ;
  for ( ; m<n ; m++)      {
    z=pt[m] ;
    PUT(trel[z],len[z])   } 
               }
          }
    /* SINGLE table */
else      {
  for (m=0 ; m<k ; m++) {
  z=pt[m] ;
  PUT(trel[z],len[z])   } 
          }

   /* PRINT TABLES OF LEN */
if (silent&4) { long p, m=0,q ;
 for (p=q=0 ; q<gpno ; p+=chsz<<1,q++) {
  if (space[p+chsz]+space[p+chsz+chsz-1]>2) {{
  printf(
  "\ntable %2d gives codelengths for move to front 1/2\n",
     q+1) ;
  for (m=p,n=0 ; n<chsz ; m++)
   x=printf("%c%3d",
   (n%10==0) ? 10 : ' ',
   space[m+chsz]), n++ ;                 }}
              }                             }
} 
   /* FILE END ? */
z=getc(fpr) ;
if (z^EOF) {
  ungetc(z,fpr) ;
  goto start ; }
if (bfct) PUT(0,8-bfct) 
}}}

      /* FILE COMPLETION */
fclose(fpr) ; fclose(fpw) ;
if (keep) unlink(argv[argcc]) ;
if (inct) x=inct ; else x=1 ;
if (silent)
  y=printf("\n file in %s  out %s", argv[argcc],chl),
  y=printf("\n  in  %%cut  out mbits %s\n",
  (silent&2) ? "tables select reps no. size" :
      " " ),
  y=printf("%6d %2d %6d %4d",
     inct,100*(inct-outct)/x,outct,8000*outct/x ) ;
if (silent &2)
  printf(" %3d %3d %1d %1d %3d\n",
    dct*8000/x,selct*1000/x,grepp,gpno,gpsz) ;
continue ;

     /* HANDLE ERRORS  */
err   : close(fpw) ; unlink(fpw) ;
err1  : close(fpr) ;
err2  : errct++ ;
fprintf(stderr,"bred3 fails, file %s\n%s\n",
   argv[argcc],error) ; 
if (errct>=errlim) return errct ;  
} /* return to ARG HANDLER */
return errct ;       /* EXIT */
}

