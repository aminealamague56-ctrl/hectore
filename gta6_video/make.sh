#!/bin/bash
# GTA 6 promo (Moroccan Darija) - 1920x1080 16:9, 30fps, 30s, H.264 + AAC stereo
# Usage: ./make.sh   -> gta6_promo_darija.mp4
set -euo pipefail
cd "$(dirname "$0")"

F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
D=30        # duration (s)
S=6         # scene length (s)

# T text start end size y color  -> text that slides up + fades in/out
T(){
  local a="if(lt(t,$2),0,if(lt(t,$2+0.4),(t-$2)/0.4,if(lt(t,$3-0.35),1,if(lt(t,$3),($3-t)/0.35,0))))"
  echo "drawtext=fontfile=$F:text_shaping=1:text='$1':fontsize=$4:fontcolor=$6:x=(w-text_w)/2:y=$5+60*max(0\,1-(t-$2)/0.4):borderw=5:bordercolor=black@0.7:shadowx=0:shadowy=8:shadowcolor=black@0.5:alpha='$a'"
}

VF="format=yuv420p"
# neon horizon lines
VF="$VF,drawbox=x=0:y=ih*0.80:w=iw:h=4:color=0xff2d95@0.8:t=fill"
VF="$VF,drawbox=x=0:y=ih*0.84:w=iw:h=2:color=0x2de2ff@0.5:t=fill"
VF="$VF,drawbox=x=0:y=ih*0.88:w=iw:h=2:color=0x2de2ff@0.3:t=fill"
VF="$VF,vignette=PI/5"
# scenes
VF="$VF,$(T 'GTA 6' 0.2 6 250 '(h-text_h)/2-140' 0xff2d95)"
VF="$VF,$(T 'قربات بزاف! وجدو راسكم' 0.7 6 90 'h/2+70' white)"
VF="$VF,$(T 'الخميس 19 نونبر 2026' 6 12 140 '(h-text_h)/2-110' 0x2de2ff)"
VF="$VF,$(T 'على PS5 و Xbox Series X|S' 6.5 12 72 'h/2+80' white)"
VF="$VF,$(T 'واش نتوما واجدين؟' 12 18 150 '(h-text_h)/2-110' 0xffd23f)"
VF="$VF,$(T 'كتبو لينا فالكومونتير شنو كتسناو' 12.5 18 72 'h/2+80' white)"
VF="$VF,$(T 'لايف مباشر نهار الخروج' 18 24 135 '(h-text_h)/2-110' 0xff2d95)"
VF="$VF,$(T 'غادي نلعبو مع بعض من الدقيقة الأولى' 18.5 24 72 'h/2+80' white)"
VF="$VF,$(T 'اشتركو فالقناة و فعلو الجرس' 24 30 115 '(h-text_h)/2-110' white)"
VF="$VF,$(T '@esjine' 24.5 30 160 'h/2+60' 0x2de2ff)"
# blinking LIVE badge during the live scene
VF="$VF,drawbox=x=80:y=70:w=230:h=90:color=0xe60023:t=fill:enable='between(t,18,24)*lt(mod(t,1),0.7)'"
VF="$VF,drawtext=fontfile=$F:text='● LIVE':fontsize=56:fontcolor=white:x=105:y=88:enable='between(t,18,24)*lt(mod(t,1),0.7)'"
# channel watermark + progress bar
VF="$VF,drawtext=fontfile=$F:text='@esjine':fontsize=40:fontcolor=white@0.75:x=w-text_w-60:y=60:enable='lt(t,24)'"
VF="$VF,drawbox=x=0:y=ih-12:w=iw*t/$D:h=12:color=0xff2d95:t=fill"
# white flash on every scene cut
VF="$VF,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.85:t=fill:enable='between(t,6,6.12)+between(t,12,12.12)+between(t,18,18.12)+between(t,24,24.12)'"
VF="$VF,fade=t=in:d=0.6,fade=t=out:st=$((D-1)):d=1"

# Synth music 128 BPM: kick, offbeat hat, bass, pad, risers before cuts, booms on cuts
B=0.46875
KICK="0.9*sin(2*PI*(45+120*exp(-25*mod(t,$B)))*mod(t,$B))*exp(-7*mod(t,$B))"
HAT="0.12*(2*random(0)-1)*exp(-45*mod(t+$B/2,$B))"
BASS="0.22*sin(2*PI*55*t)*(1-exp(-12*mod(t,$B)))*lt(mod(t,4*$B),3*$B)+0.22*sin(2*PI*65.4*t)*(1-exp(-12*mod(t,$B)))*gte(mod(t,4*$B),3*$B)"
PAD="0.05*(sin(2*PI*220*t)+sin(2*PI*261.6*t)+sin(2*PI*329.6*t))"
RISE="0.25*(2*random(1)-1)*gt(mod(t,$S),$S-1.2)*pow((mod(t,$S)-($S-1.2))/1.2,2)*lt(t,$D-1)"
BOOM="0.8*sin(2*PI*38*mod(t,$S))*exp(-3*mod(t,$S))*gt(t,1)"
EXPR="$KICK+$HAT+$BASS+$PAD+$RISE+$BOOM"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "gradients=s=1920x1080:c0=0x1a0033:c1=0x3a0a4a:c2=0x0b1d3a:x0=0:y0=0:x1=1920:y1=1080:speed=0.02:d=$D:r=30" \
  -f lavfi -i "aevalsrc='$EXPR|$EXPR':s=48000:c=stereo:d=$D" \
  -vf "$VF" \
  -af "alimiter=limit=0.9,loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.5,afade=t=out:st=$((D-2)):d=2,aresample=48000" \
  -c:v libx264 -preset slow -profile:v high -pix_fmt yuv420p -crf 18 -r 30 -g 15 -bf 2 \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart -t $D \
  gta6_promo_darija.mp4

echo "OK: $(pwd)/gta6_promo_darija.mp4"
