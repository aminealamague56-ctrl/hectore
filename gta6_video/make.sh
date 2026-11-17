#!/bin/bash
# Promo video GTA 6 - Darija - 1920x1080, 30s
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
T(){ # text start end size y color
 echo "drawtext=fontfile=$F:text_shaping=1:text='$1':fontsize=$4:fontcolor=$6:x=(w-text_w)/2:y=$5:borderw=4:bordercolor=black@0.6:alpha='if(lt(t,$2),0,if(lt(t,$2+0.5),(t-$2)/0.5,if(lt(t,$3-0.5),1,if(lt(t,$3),($3-t)/0.5,0))))'"
}
VF="format=yuv420p"
VF="$VF,drawbox=x=0:y=ih-12:w=iw*t/30:h=12:color=0xff2d95:t=fill"
VF="$VF,$(T 'GTA 6' 0 6 230 '(h-text_h)/2-120' 0xff2d95)"
VF="$VF,$(T 'جاية! بقاو غير جوج أيام' 0.6 6 90 '(h)/2+80' white)"
VF="$VF,$(T '19 نونبر 2026' 6 12 170 '(h-text_h)/2-90' 0x2de2ff)"
VF="$VF,$(T 'PS5 و Xbox Series X|S' 6.5 12 80 '(h)/2+90' white)"
VF="$VF,$(T 'واش نتوما واجدين؟' 12 18 140 '(h-text_h)/2-90' 0xffd23f)"
VF="$VF,$(T 'كتبو لينا فالكومونتير شنو كتسناو' 12.5 18 70 '(h)/2+90' white)"
VF="$VF,$(T 'لايف مباشر نهار الخروج' 18 24 130 '(h-text_h)/2-90' 0xff2d95)"
VF="$VF,$(T 'غادي نلعبو مع بعض من الدقيقة الأولى' 18.5 24 70 '(h)/2+90' white)"
VF="$VF,$(T 'اشتركو فالقناة و فعلو الجرس' 24 30 110 '(h-text_h)/2-90' white)"
VF="$VF,$(T '@esjine' 24.5 30 150 '(h)/2+80' 0x2de2ff)"
ffmpeg -y -f lavfi -i "gradients=s=1920x1080:c0=0x1a0033:c1=0x3a0a4a:c2=0x0b1d3a:x0=0:y0=0:x1=1920:y1=1080:speed=0.02:d=30:r=30" \
 -f lavfi -i "aevalsrc='0.5*sin(2*PI*55*t)*exp(-8*mod(t,0.5))+0.15*sin(2*PI*220*t)*exp(-20*mod(t+0.25,0.5))':s=44100:d=30" \
 -vf "$VF" -af "afade=t=in:d=1,afade=t=out:st=28.5:d=1.5" -c:v libx264 -preset medium -crf 20 -c:a aac -b:a 160k -shortest gta6_promo_darija.mp4
ffmpeg -y -ss 7 -i gta6_promo_darija.mp4 -frames:v 1 thumb_check.png 2>/dev/null
