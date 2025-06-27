/*=============================================================================
                                                           
 d8b 888b     d888 888b     d888 8888888888 8888888b.   .d8888b.  8888888888 
 Y8P 8888b   d8888 8888b   d8888 888        888   Y88b d88P  Y88b 888        
     88888b.d88888 88888b.d88888 888        888    888 Y88b.      888        
 888 888Y88888P888 888Y88888P888 8888888    888   d88P  "Y888b.   8888888    
 888 888 Y888P 888 888 Y888P 888 888        8888888P"      "Y88b. 888        
 888 888  Y8P  888 888  Y8P  888 888        888 T88b         "888 888        
 888 888   "   888 888   "   888 888        888  T88b  Y88b  d88P 888        
 888 888       888 888       888 8888888888 888   T88b  "Y8888P"  8888888888                                                                 
                                                                            
    Copyright (c) Pascal Gilcher. All rights reserved.
    
    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

===============================================================================

    Launchpad is a prepass effect that prepares various data to use 
	in later shaders.

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

	This contains only the older optical flow code, which was far better for
	motion based visual effects (like motion blur)

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef LAUNCHPAD_DEBUG_OUTPUT
 #define LAUNCHPAD_DEBUG_OUTPUT 	  	0		//[0 or 1] 1: enables debug output of the motion vectors
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform int OPTICAL_FLOW_RES <
	ui_type = "combo";
    ui_label = "Flow Resolution";
	ui_items = "Quarter Resolution\0Half Resolution\0Full Resolution\0";
	ui_tooltip = "Higher resolution vectors are more accurate but cost more performance.";
    ui_category = "Motion Estimation / Optical Flow";
> = 0;

uniform int OPTICAL_FLOW_Q <
	ui_type = "combo";
    ui_label = "Flow Quality";
	ui_items = "Low\0Medium\0High\0";
	ui_tooltip = "Higher settings produce more accurate results, at a performance cost.";
	ui_category = "Motion Estimation / Optical Flow";
> = 0;

#if LAUNCHPAD_DEBUG_OUTPUT != 0
uniform int DEBUG_MODE < 
    ui_type = "combo";
	ui_items = "Optical Flow Vectors\0Optical Flow\0";
	ui_label = "Debug Output";
> = 0;
#endif

uniform int UIHELP <
	ui_type = "radio";
	ui_label = " ";	
	ui_text ="\nDescription for preprocessor definitions:\n"
	"\n"
	"LAUNCHPAD_DEBUG_OUTPUT\n"
	"\n"
	"Various debug outputs\n"
	"0: off\n"
	"1: on\n";
	ui_category_closed = false;
>;

/*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF4 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF5 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF6 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF7 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform bool debug_key_down < source = "key"; keycode = 0x46; mode = ""; >;

uniform bool USE_SIMPLE_MIP_PYRAMID <  > = false;
*/

/*=============================================================================
	Textures, Samplers, Globals, Structs
=============================================================================*/

//do NOT change anything here. "hurr durr I changed this and now it works"
//you ARE breaking things down the line, if the shader does not work without changes
//here, it's by design.

texture ColorInputTex : COLOR;
texture DepthInputTex : DEPTH;
sampler ColorInput 	{ Texture = ColorInputTex; };
sampler DepthInput  { Texture = DepthInputTex; };

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_depth.fxh"
#include ".\MartysMods\mmx_math.fxh"
#include ".\MartysMods\mmx_qmc.fxh"
#include ".\MartysMods\mmx_camera.fxh"
#include ".\MartysMods\mmx_texture.fxh"

namespace Velocity
{
	//motion vectors, RGBA16F, XY = delta uv, Z = confidence, W = depth because why not
	texture OldMotionVectorsTex        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F;     };
	sampler sOldMotionVectorsTex       { Texture = OldMotionVectorsTex; };

	float2 get_motion(float2 uv)
	{
		return tex2Dlod(sOldMotionVectorsTex, uv, 0).xy;
	}

	float4 get_motion_wide(float2 uv)
	{
		return tex2Dlod(sOldMotionVectorsTex, uv, 0);
	}
}

uniform uint FRAMECOUNT < source = "framecount"; >;
uniform float FRAMETIME < source = "frametime"; >;

texture OldMotionTexNewA       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA32F; MipLevels = 5;};
sampler sOldMotionTexNewA      { Texture = OldMotionTexNewA;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture OldMotionTexNewB       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA32F; MipLevels = 5;};
sampler sOldMotionTexNewB      { Texture = OldMotionTexNewB;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

texture OldMotionTexUpscale    { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;};
sampler sOldMotionTexUpscale   { Texture = OldMotionTexUpscale;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture OldMotionTexUpscale2   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;};
sampler sOldMotionTexUpscale2  { Texture = OldMotionTexUpscale2;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

#define OldMotionTexIntermediateTex0 			Velocity::OldMotionVectorsTex
#define sOldMotionTexIntermediateTex0 			Velocity::sOldMotionVectorsTex

//curr in x, prev in y
texture OldFeaturePyramidLevel0   { Width = BUFFER_WIDTH;   	  Height = BUFFER_HEIGHT;        Format = RG8; };
storage stOldFeaturePyramidLevel0  { Texture = OldFeaturePyramidLevel0;};
texture OldFeaturePyramidLevel1   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RG16F;};
texture OldFeaturePyramidLevel2   { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RG16F;};
texture OldFeaturePyramidLevel3   { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RG16F;};
texture OldFeaturePyramidLevel4   { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RG16F;};
texture OldFeaturePyramidLevel5   { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RG16F;};
texture OldFeaturePyramidLevel6   { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RG16F;};
texture OldFeaturePyramidLevel7   { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = RG16F;};
sampler sOldFeaturePyramidLevel0  { Texture = OldFeaturePyramidLevel0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sOldFeaturePyramidLevel1  { Texture = OldFeaturePyramidLevel1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sOldFeaturePyramidLevel2  { Texture = OldFeaturePyramidLevel2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sOldFeaturePyramidLevel3  { Texture = OldFeaturePyramidLevel3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sOldFeaturePyramidLevel4  { Texture = OldFeaturePyramidLevel4; AddressU = MIRROR; AddressV = MIRROR; };
sampler sOldFeaturePyramidLevel5  { Texture = OldFeaturePyramidLevel5; AddressU = MIRROR; AddressV = MIRROR; };
sampler sOldFeaturePyramidLevel6  { Texture = OldFeaturePyramidLevel6; AddressU = MIRROR; AddressV = MIRROR; };
sampler sOldFeaturePyramidLevel7  { Texture = OldFeaturePyramidLevel7; AddressU = MIRROR; AddressV = MIRROR; };

texture OldDepthLowresPacked          { Width = BUFFER_WIDTH/3;   Height = BUFFER_HEIGHT/3;   Format = RG16F; };
sampler sOldDepthLowresPacked         { Texture = OldDepthLowresPacked; MipFilter=POINT; MagFilter=POINT; MinFilter=POINT;}; 

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         //XYZ idx of thread inside group
    uint3 groupid           : SV_GroupID;               //XYZ idx of group inside dispatch
    uint3 dispatchthreadid  : SV_DispatchThreadID;      //XYZ idx of thread inside dispatch
    uint threadid           : SV_GroupIndex;            //flattened idx of thread inside group
};

static float2 star_kernel[13] = 
{
	float2(0, 0),
	//inner ring
	float2(-1, -2),
	float2(1, -2),
	float2(2, 0),
	float2(1, 2),
	float2(-1, 2),
	float2(-2, 0),
	//outer ring
	float2(-3, -2),
	float2(0,-4),
	float2(3, -2),
	float2(3, 2),
	float2(0, 4),
	float2(-3, 2)
};


/*=============================================================================
	Functions
=============================================================================*/

float get_prev_depth(float2 uv)
{
	return tex2Dlod(sOldDepthLowresPacked, uv, 0).y;
}

float2 downsample_feature(sampler s, float2 uv)
{
	float2 res = 0;	
	float2 texelsize = rcp(tex2Dsize(s));	
	float wsum = 0;
#if 0
	for(int x = 0; x < 6; x++)
	for(int y = 0; y < 6; y++)
	{
		float2 offs = float2(x, y); //0 to 5
		offs -= 2.5; // -2.5 to 2.5
		float g = exp(-dot(offs, offs) * 0.1);
		res += g * tex2D(s, uv + offs * texelsize).rg;
		wsum += g;
	}
#else
	[unroll]for(int x = -1; x <= 1; x++)
	[unroll]for(int y = -1; y <= 1; y++)
	{
		float2 offs = float2(x, y) * 2;

		float2 offs_tl = offs + float2(-0.5, -0.5);
		float2 offs_tr = offs + float2( 0.5, -0.5);
		float2 offs_bl = offs + float2(-0.5,  0.5);
		float2 offs_br = offs + float2( 0.5,  0.5);

		float4 g;
		g.x = dot(offs_tl, offs_tl);
		g.y = dot(offs_tr, offs_tr);
		g.z = dot(offs_bl, offs_bl);
		g.w = dot(offs_br, offs_br);
		g = exp(-g*0.1);
		float tg = dot(g, 1);
		offs = (offs_tl * g.x + offs_tr * g.y + offs_bl * g.z + offs_br * g.w) / tg;
	

		//float tg = exp(-dot(offs, offs) * 0.1);
		res += tg * tex2Dlod(s, uv + offs * texelsize, 0).rg;
		wsum += tg;
	}
#endif
	

	return res / wsum;	
}

void DownsampleFeaturePS1(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel0, i.uv);} 
void DownsampleFeaturePS2(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel1, i.uv);} 
void DownsampleFeaturePS3(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel2, i.uv);} 
void DownsampleFeaturePS4(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel3, i.uv);} 
void DownsampleFeaturePS5(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel4, i.uv);} 
void DownsampleFeaturePS6(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel5, i.uv);}
void DownsampleFeaturePS7(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sOldFeaturePyramidLevel6, i.uv);}

float loss(float a, float b)
{
	float t = a - b;
	return abs(t); //SAD
}

float3 loss_grad(float3 a, float b)
{
	float3 t = a - b;
	return abs(t); //SAD
}

struct AdamOptimizer
{
	float2 m;
	float v;
	float beta1decayed, beta2decayed;
	float beta1, beta2, epsilon;
	float lr;
};

AdamOptimizer init_adam(int T)
{
	AdamOptimizer a;
	a.m = a.v = 0;
	a.beta1decayed = a.beta2decayed = 1;

	a.epsilon = 0.00000001;
	a.beta1 = 0.9;
	a.beta2 = 0.999;
	a.lr = 0.000625;
	return a;
}

float2 update_adam(inout AdamOptimizer a, float2 grad)
{
	float2 g = grad;
	a.m = lerp(g, a.m, a.beta1);
	a.v = lerp(dot(g, g), a.v, a.beta2);

	a.beta1decayed *= a.beta1;
	a.beta2decayed *= a.beta2;

	{
		a.beta1decayed = 0;
		a.beta2decayed = 0;
	}

	float2 mhat = a.m / (1 - a.beta1decayed);
	float vhat  = a.v / (1 - a.beta2decayed);

	mhat *= 0.2;
	//return a.lr * mhat / (sqrt(vhat) + a.epsilon);
	return a.lr * (mhat * rsqrt(max(vhat, a.epsilon)));
}

float4 gradient_block_matching_new(sampler s_feature, sampler s_flow, float2 uv, int level, const int blocksize)
{	
	float2 texelsize = rcp(tex2Dsize(s_feature));
	float2 search_scale = texelsize;

	float level_fi = float(level / 7.0); //0 to 1

	int num_steps = level < 2 ? 4 : 8;
	num_steps *= 1 + OPTICAL_FLOW_Q;

	float2 deltax = texelsize * float2(0.01, 0);
	float2 deltay = texelsize * float2(0, 0.01);

	//get local block data
	float local_block[13];

	[unroll]
	for(uint k = 0; k < blocksize; k++) //always fetch it completely
	{
		float2 tuv = uv + star_kernel[k] * search_scale;
		local_block[k] = tex2Dlod(s_feature, tuv, 0).x;
	}	

	float4 coarse_layer = 0;//tex2D(s_flow, uv);	

	//if we're not the first pass, do some neighbour pooling to get a better initial guess
	[branch]
	if(level < 7)
	{
		coarse_layer = tex2Dlod(s_flow, uv, 0);
		float best_sad = 0;

		[unroll]
		for(uint k = 0; k < blocksize; k++)
		{
			float2 tuv = uv + coarse_layer.xy + star_kernel[k] * search_scale;
			best_sad += loss(local_block[k], tex2Dlod(s_feature, tuv, 0).y);	
		}

		float2 motion_texelsize = rcp(tex2Dsize(s_flow));	
		motion_texelsize = max(motion_texelsize, texelsize);
		int2 sector_offs[4] = {int2(-1, -2), int2(-2, 0), int2(1, -1), int2(0, 1)};
		[unroll]
		for(int sec = 0; sec < 4; sec++)
		{
			float2 flows[4];
			flows[0] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(0, 0)), 0).xy;
			flows[1] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(1, 0)), 0).xy;
			flows[2] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(0, 1)), 0).xy;
			flows[3] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(1, 1)), 0).xy;

			float3 median = float3(0, 0, 1e10);
			
			[unroll]for(int j = 0; j < 4; j++)
			{
				float diffsum = 0;
				diffsum += distance(flows[j], flows[0]);
				diffsum += distance(flows[j], flows[1]);
				diffsum += distance(flows[j], flows[2]);
				diffsum += distance(flows[j], flows[3]);

				median = diffsum < median.z ? float3(flows[j], diffsum) : median;	
			}

			median.z = 0; //now loss

			[loop]
			for(uint k = 0; k < blocksize; k++)
			{
				float2 tuv = uv + median.xy + star_kernel[k] * search_scale;
				median.z += loss(local_block[k], tex2Dlod(s_feature, tuv, 0).y);
				if(median.z > best_sad) break;
			}

			[branch]
			if(median.z < best_sad)
			{
				best_sad = median.z;
				coarse_layer.xy = median.xy;
			}			
		}
	}
	
	//once found, proceed
	float2 total_motion = coarse_layer.xy;
	
	float3 SAD = 0; //center, +dx, +dy
	float2 texturesize = tex2Dsize(s_feature);	

	//read local gradient
	[unroll]
	for(uint k = 0; k < blocksize; k++)
	{		
		float2 tuv = uv + star_kernel[k] * search_scale;
		float g = local_block[k];
		float f;
		f = tex2Dlod(s_feature, tuv + total_motion,          0).y;
		SAD.x += loss(f, g);
		f = tex2Dlod(s_feature, tuv + total_motion + deltax, 0).y;		
		SAD.y += loss(f, g);
		f = tex2Dlod(s_feature, tuv + total_motion + deltay, 0).y;
		SAD.z += loss(f, g);
    }

	float2 grad = (SAD.yz - SAD.x) / float2(deltax.x, deltay.y);
	AdamOptimizer adam = init_adam(num_steps);

	float2 local_motion = 0;
	float2 best_local_motion = 0;
	float  best_SAD = SAD.x;
	adam.lr *= 1.0 + level;	
	adam.lr *= 0.5;
	adam.lr /= 1.0 + OPTICAL_FLOW_Q;
	float2 local_motion_prev = local_motion;
	
	int fails = 0;
	int max_fails = 4 * (1 + OPTICAL_FLOW_Q);
	[loop]
	while(num_steps-- >= 0 && fails < max_fails)
	{		
		//nesterov momentum
		float2 curr_grad_step = update_adam(adam, grad);

		if(maxc(abs(curr_grad_step) * BUFFER_SCREEN_SIZE) < 0.1) 
			break;

		local_motion = local_motion_prev - curr_grad_step;
		local_motion_prev = local_motion;
	
		local_motion -= curr_grad_step;//look ahead using curr gradient
		SAD = 0;

		[unroll]
		for(uint k = 0; k < blocksize; k++)
		{
			float2 tuv = uv + total_motion + local_motion + star_kernel[k] * search_scale;
			float g = local_block[k];

			float f;	
			f = tex2Dlod(s_feature, tuv, 0).y;	
			SAD.x += loss(f, g);
			f = tex2Dlod(s_feature, tuv + deltax, 0).y;
			SAD.y += loss(f, g);
			f = tex2Dlod(s_feature, tuv + deltay, 0).y;
			SAD.z += loss(f, g);
		}		

		[flatten]
		if(SAD.x < best_SAD)
		{
			best_SAD = SAD.x;
			best_local_motion = local_motion;
			fails = 0;
		}
		else 
		{
			fails++;
		}
		
		grad = (SAD.yz - SAD.x) / float2(deltax.x, deltay.y);		
	}

	local_motion = best_local_motion;
	total_motion += local_motion;

	float prev_depth_at_motion = get_prev_depth(uv + total_motion);
	float4 curr_layer = float4(total_motion, prev_depth_at_motion, best_SAD);
	return curr_layer;
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(log(1 + dist * 1000.0  /* / FRAMETIME */)));//normalize by frametime such that we don't need to adjust visualization intensity all the time
}

//turbo colormap fit, turned into MADD form
float3 gradient(float t)
{	
	t = saturate(t);
	float3 res = float3(59.2864, 2.82957, 27.3482);
	res = mad(res, t.xxx, float3(-152.94239396, 4.2773, -89.9031));	
	res = mad(res, t.xxx, float3(132.13108234, -14.185, 110.36276771));
	res = mad(res, t.xxx, float3(-42.6603, 4.84297, -60.582));
	res = mad(res, t.xxx, float3(4.61539, 2.19419, 12.6419));
	res = mad(res, t.xxx, float3(0.135721, 0.0914026, 0.106673));
	return saturate(res);
}

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
    return o;
}
/*
texture2D StateCounterTex	{ Format = R32F;  	};
sampler2D sStateCounterTex	{ Texture = StateCounterTex;  };

float4 FrameWriteVS(in uint id : SV_VertexID) : SV_Position {return float4(!debug_key_down, !debug_key_down, 0, 1);}
float  FrameWritePS(in float4 vpos : SV_Position) : SV_Target0 {return FRAMECOUNT;}
*/
void WriteDepthFeaturePS(in VSOUT i, out float2 o : SV_Target0)
{
	o = Depth::get_linear_depth(i.uv);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x + 1) discard;
}

void WriteFeaturePS(in VSOUT i, out float4 o : SV_Target0)
{	
	o = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x + 1) discard;
}

void WritePrevLowresDepthPS(in VSOUT i, out float2 o : SV_Target0)
{
	o = Depth::get_linear_depth(i.uv);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void WriteFeaturePS2(in VSOUT i, out float4 o : SV_Target0)
{	
	o = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void BlockMatchingPassPS8(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel7, sOldMotionTexNewB, i.uv, 7, 7);}
void BlockMatchingPassPS7(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel6, sOldMotionTexNewA, i.uv, 6, 7);}
void BlockMatchingPassPS6(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel5, sOldMotionTexNewB, i.uv, 5, 7);}
void BlockMatchingPassPS5(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel4, sOldMotionTexNewA, i.uv, 4, 7);}
void BlockMatchingPassPS4(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel3, sOldMotionTexNewB, i.uv, 3, 7);}
void BlockMatchingPassPS3(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel2, sOldMotionTexNewA, i.uv, 2, 7);}
void BlockMatchingPassPS2(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel1, sOldMotionTexNewB, i.uv, 1, 13);}
void BlockMatchingPassPS1(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sOldFeaturePyramidLevel0, sOldMotionTexNewA, i.uv, 0, 13);}

void UpscaleFlowPS0(in VSOUT i, out float4 o : SV_Target0)
{
	if(OPTICAL_FLOW_RES < 1) 
		discard;
	o = gradient_block_matching_new(sOldFeaturePyramidLevel0, sOldMotionTexNewB, i.uv, 0, 13);
}

void UpscaleFlowPS1(in VSOUT i, out float4 o : SV_Target0)
{	
	if(OPTICAL_FLOW_RES < 2) 
		discard;
	o = gradient_block_matching_new(sOldFeaturePyramidLevel0, sOldMotionTexUpscale, i.uv, 0, 13);
}

void CopyToFullres(in VSOUT i, out float4 o : SV_Target0)
{
	o = 0;
	if(OPTICAL_FLOW_RES == 0)      o = tex2Dlod(sOldMotionTexNewB, i.uv, 0);
	else if(OPTICAL_FLOW_RES == 1) o = tex2Dlod(sOldMotionTexUpscale, i.uv, 0);
	else if(OPTICAL_FLOW_RES == 2) o = tex2Dlod(sOldMotionTexUpscale2, i.uv, 0);
}

#if LAUNCHPAD_DEBUG_OUTPUT != 0
void DebugPS(in VSOUT i, out float3 o : SV_Target0)
{	
	o = 0;
	switch(DEBUG_MODE)
	{
		case 0:
		{
			float2 tile_size = 16.0;
			float2 tile_uv = i.uv * BUFFER_SCREEN_SIZE / tile_size;
			float2 motion = Velocity::get_motion((floor(tile_uv) + 0.5) * tile_size * BUFFER_PIXEL_SIZE);

			float3 chroma = showmotion(motion);
			
			motion *= BUFFER_SCREEN_SIZE;
			float velocity = length(motion);
			float2 mainaxis = velocity == 0 ? 0 : motion / velocity;
			float2 otheraxis = float2(mainaxis.y, -mainaxis.x);
			float2x2 rotation = float2x2(mainaxis, otheraxis);

			tile_uv = (frac(tile_uv) - 0.5) * tile_size;
			tile_uv = mul(tile_uv, rotation);
			o = tex2Dlod(ColorInput, i.uv, 0).rgb;
			float mask = smoothstep(min(velocity, 2.5), min(velocity, 2.5) - 1, abs(tile_uv.y)) * smoothstep(velocity, velocity - 1.0, abs(tile_uv.x));

			o = lerp(o, chroma, mask);
			break;
		}
		case 1: o = showmotion(Velocity::get_motion(i.uv)); break;
	}	
}
#endif

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_Velocity
<
    ui_label = "iMMERSE: Velocity (enable and move to the top!)";
    ui_tooltip =        
        "                           MartysMods - Velocity                             \n"
        "                   MartysMods Epic ReShade Effects (iMMERSE)                  \n"
        "______________________________________________________________________________\n"
        "\n"

        "This is an older version of Launchpad motion vectors better suited for motion \n"
		"blur. This can now co-exist with newer versions of Launchpad.                 \n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                            \n"
        "\n"       
        "______________________________________________________________________________";
>
{
	pass {VertexShader = MainVS;PixelShader = WriteDepthFeaturePS;  RenderTarget0 = OldDepthLowresPacked; RenderTargetWriteMask = 1 << 0;} 
    pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; 	    RenderTarget0 = OldFeaturePyramidLevel0; RenderTargetWriteMask = 1 << 0;} 
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS1;	RenderTarget = OldFeaturePyramidLevel1;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS2;	RenderTarget = OldFeaturePyramidLevel2;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS3;	RenderTarget = OldFeaturePyramidLevel3;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS4;	RenderTarget = OldFeaturePyramidLevel4;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS5;	RenderTarget = OldFeaturePyramidLevel5;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS6;	RenderTarget = OldFeaturePyramidLevel6;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS7;	RenderTarget = OldFeaturePyramidLevel7;}

	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS8;	RenderTarget = OldMotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS7;	RenderTarget = OldMotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS6;	RenderTarget = OldMotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS5;	RenderTarget = OldMotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS4;	RenderTarget = OldMotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS3;	RenderTarget = OldMotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS2;	RenderTarget = OldMotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS1;	RenderTarget = OldMotionTexNewB;}

	pass {VertexShader = MainVS;PixelShader = UpscaleFlowPS0;		RenderTarget = OldMotionTexUpscale;}
	pass {VertexShader = MainVS;PixelShader = UpscaleFlowPS1;		RenderTarget = OldMotionTexUpscale2;}
		
	pass {VertexShader = MainVS;PixelShader = CopyToFullres;		RenderTarget = OldMotionTexIntermediateTex0;}

	pass {VertexShader = MainVS;PixelShader = WritePrevLowresDepthPS; RenderTarget0 = OldDepthLowresPacked; RenderTargetWriteMask = 1 << 1;} 
	pass {VertexShader = MainVS;PixelShader = WriteFeaturePS2; RenderTarget0 = OldFeaturePyramidLevel0; RenderTargetWriteMask = 1 << 1;}	

#if LAUNCHPAD_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }			
#endif
}
