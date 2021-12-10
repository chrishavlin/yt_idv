in vec2 UV;

out vec4 color;

void main(){
   float scaled = 0;
   if (use_db) {
      scaled = texture(db_tex, UV).x;
   } else {
      scaled = texture(fb_tex, UV).x;
   }
   float alpha = texture(fb_tex, UV).a;
   if (alpha == 0.0) discard;
   float cm = cmap_min;
   float cp = cmap_max;

   if (cmap_log > 0.5) {
       float inv_log10 = 1 / log(10);  // log is natural log, need to convert to base10
       scaled = log(scaled) * inv_log10;
       cm = log(cm) * inv_log10;
       cp = log(cp) * inv_log10;
   }
   color = texture(cm_tex, (scaled - cm) / (cp - cm));
   gl_FragDepth = texture(db_tex, UV).r;
}
