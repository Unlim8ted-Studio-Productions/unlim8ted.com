// Content-addressed, integrity-checked Unity file reconstruction for static hosts.
export function chooseVariant(file){return file.variants.find(v=>v.encoding==='gzip' && typeof DecompressionStream!=='undefined')||file.variants.find(v=>v.encoding==='identity');}
export async function reconstruct(file,variant,{base=document.baseURI,onBytes=()=>{},signal}={}){
 const buffers=new Array(variant.parts.length);let cursor=0;
 async function worker(){
  while(cursor<variant.parts.length){const index=cursor++,part=variant.parts[index];const response=await fetch(new URL(part.url,base),{signal,cache:'force-cache'});
   if(!response.ok)throw new Error(`Could not download ${file.name} (${response.status}). Try again when your connection is ready.`);
   const pieces=[];let length=0;
   if(response.body){const reader=response.body.getReader();while(true){const {done,value}=await reader.read();if(done)break;pieces.push(value);length+=value.length;onBytes(value.length);}}
   else {const data=new Uint8Array(await response.arrayBuffer());pieces.push(data);length=data.length;onBytes(length);}
   if(length!==part.bytes)throw new Error(`Incomplete download of ${file.name}. Please retry.`);
   const joined=new Uint8Array(length);let offset=0;for(const piece of pieces){joined.set(piece,offset);offset+=piece.length;}
   if(globalThis.crypto?.subtle){const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',joined))].map(x=>x.toString(16).padStart(2,'0')).join('');if(hash!==part.sha256)throw new Error(`The ${file.name} download failed its integrity check. Please retry.`);}
   buffers[index]=joined;
  }
 }
 await Promise.all([worker(),worker()]);
 let blob=new Blob(buffers,{type:file.mime});
 if(variant.encoding==='gzip')blob=await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip')),{headers:{'Content-Type':file.mime}}).blob();
 if(blob.size!==file.bytes)throw new Error(`The reconstructed ${file.name} has an unexpected size.`);
 return URL.createObjectURL(blob);
}
