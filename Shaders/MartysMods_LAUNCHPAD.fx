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

uniform int OPTICAL_FLOW_Q <
	ui_type = "combo";
    ui_label = "Flow Quality";
	ui_items = "Low\0Medium\0High\0";
	ui_tooltip = "Higher settings produce more accurate results, at a performance cost.";
	ui_category = "Motion Estimation / Optical Flow";
> = 0;

uniform int OPTICAL_FLOW_OPT <
	ui_type = "combo";
    ui_label = "Flow Optimizer";
	ui_items = "Sophia (Second-Order Hessian Optimizer)\0Newton\0";
	ui_tooltip = "Launchpad's Optical Flow uses gradient descent, similar to AI training.\n\n"
				 "Sophia converges better at high quality settings.\n"
				 "Newton descents faster at low settings but may converge worse.";
	ui_category = "Motion Estimation / Optical Flow";
> = 0;

uniform bool ENABLE_SMOOTH_NORMALS <	
	ui_label = "Enable Smooth Normals";
	ui_tooltip = "Filters the normal buffer to reduce low-poly look in MXAO and RTGI."
	"\n\n"
	"Lighting algorithms depend on normal vectors, which describe the orientation\n"
	"of the geometry in the scene. As ReShade does not access the game's own normals,\n"
	"they are generated from the depth buffer instead. However, this process is lossy\n"
	"and does not contain normal maps and smoothing groups.\n"
	"As a result, they represent the true (blocky) object shapes and lighting calculated\n"
	"using them can make the low-poly appearance of geometry apparent.\n";
	ui_category = "NORMAL MAPS";	
> = false;

uniform bool ENABLE_TEXTURED_NORMALS <	
	ui_label = "Enable Texture Normals";
	ui_tooltip = "Estimates surface relief based on color information, for more accurate geometry representation.\n"
	             "Requires smooth normals to be enabled!";	
	ui_category = "NORMAL MAPS";	
> = false;

uniform float TEXTURED_NORMALS_RADIUS <
	ui_type = "drag";
	ui_label = "Textured Normals Sample Radius";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_category = "NORMAL MAPS";	
> = 0.5;

uniform float TEXTURED_NORMALS_INTENSITY <
	ui_type = "drag";
	ui_label = "Textured Normals Intensity";
	ui_tooltip = "Higher values cause stronger surface bumpyness.";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_category = "NORMAL MAPS";	
> = 0.5;

uniform int TEXTURED_NORMALS_QUALITY <
	ui_type = "slider";
	ui_min = 1; ui_max = 3;
    ui_label = "Textured Normals Quality";
    ui_tooltip = "Higher settings produce more accurate results, at a performance cost.";
    ui_category = "NORMAL MAPS";	
> = 2;

#if LAUNCHPAD_DEBUG_OUTPUT != 0
uniform int DEBUG_MODE < 
    ui_type = "combo";
	ui_items = "All\0Optical Flow\0Optical Flow Vectors\0Normals\0Depth\0";
	ui_label = "Debug Output";
	ui_category = "Debug";
> = 0;
#endif


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

uniform bool DISABLE_POOLING <  > = false;
uniform bool DISABLE_UPSCALING <  > = false;
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
#include ".\MartysMods\mmx_deferred.fxh"
#include ".\MartysMods\mmx_texture.fxh"
#include ".\MartysMods\mmx_random.fxh"

uniform uint FRAMECOUNT < source = "framecount"; >;
uniform float FRAMETIME < source = "frametime"; >;

texture MotionTexNewA       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexNewA      { Texture = MotionTexNewA;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexNewB       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexNewB      { Texture = MotionTexNewB;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexUpscale    { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;};
sampler sMotionTexUpscale   { Texture = MotionTexUpscale;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexUpscale2   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;};
sampler sMotionTexUpscale2  { Texture = MotionTexUpscale2;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

//Yes I know you like to optimize blue noise away in favor for some shitty PRNG function, don't.
texture BlueNoiseJitterTex     < source = "iMMERSE_bluenoise.png"; > { Width = 32; Height = 32; Format = RGBA8; };
sampler	sBlueNoiseJitterTex   { Texture = BlueNoiseJitterTex; AddressU = WRAP; AddressV = WRAP; };

texture FlowFeaturesCurrL0   { Width = BUFFER_WIDTH >> 0;   Height = BUFFER_HEIGHT >> 0;   Format = R8; };
texture FlowFeaturesCurrL1   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = R16F;};
texture FlowFeaturesCurrL2   { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = R16F;};
texture FlowFeaturesCurrL3   { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = R16F;};
texture FlowFeaturesCurrL4   { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = R16F;};
texture FlowFeaturesCurrL5   { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = R16F;};
texture FlowFeaturesCurrL6   { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = R16F;};
texture FlowFeaturesCurrL7   { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = R16F;};
sampler sFlowFeaturesCurrL0  { Texture = FlowFeaturesCurrL0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sFlowFeaturesCurrL1  { Texture = FlowFeaturesCurrL1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL2  { Texture = FlowFeaturesCurrL2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL3  { Texture = FlowFeaturesCurrL3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL4  { Texture = FlowFeaturesCurrL4; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL5  { Texture = FlowFeaturesCurrL5; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL6  { Texture = FlowFeaturesCurrL6; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL7  { Texture = FlowFeaturesCurrL7; AddressU = MIRROR; AddressV = MIRROR; };
texture FlowFeaturesPrevL0   { Width = BUFFER_WIDTH >> 0;   Height = BUFFER_HEIGHT >> 0;   Format = R8; };
texture FlowFeaturesPrevL1   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = R16F;};
texture FlowFeaturesPrevL2   { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = R16F;};
texture FlowFeaturesPrevL3   { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = R16F;};
texture FlowFeaturesPrevL4   { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = R16F;};
texture FlowFeaturesPrevL5   { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = R16F;};
texture FlowFeaturesPrevL6   { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = R16F;};
texture FlowFeaturesPrevL7   { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = R16F;};
sampler sFlowFeaturesPrevL0  { Texture = FlowFeaturesPrevL0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sFlowFeaturesPrevL1  { Texture = FlowFeaturesPrevL1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL2  { Texture = FlowFeaturesPrevL2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL3  { Texture = FlowFeaturesPrevL3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL4  { Texture = FlowFeaturesPrevL4; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL5  { Texture = FlowFeaturesPrevL5; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL6  { Texture = FlowFeaturesPrevL6; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL7  { Texture = FlowFeaturesPrevL7; AddressU = MIRROR; AddressV = MIRROR; };

//miplevel 3 is copied to previous frame!
//in theory I should be computing the optical flow at the lower TAAU resolution. Maybe later.
texture LinearDepthCurr      { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F; MipLevels = 4; };
sampler sLinearDepthCurr     { Texture = LinearDepthCurr; }; 
texture LinearDepthPrevLo      { Width = BUFFER_WIDTH>>3;   Height = BUFFER_HEIGHT>>3;   Format = R16F; };
sampler sLinearDepthPrevLo     { Texture = LinearDepthPrevLo; };

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


//
//          A
//
//	  7   1   4   8
//
//		6	0   2
// 
//    C   3   6   B
//
//          9

//this pattern allows to use isotropic kernels with 4, 7, 10, 13, 16 and 19 samples
static float2 star_kernel[19] = 
{
	//center
	float2(0, 0),
	//inner ring first 3
	float2(-1, 2),
	float2(2, 0),
	float2(-1, -2),	
	//inner ring second 3
	float2(1, 2),
	float2(1, -2),
	float2(-2, 0),
	//outer ring, first 3
	float2(-3, 2),
	float2(3, 2),
	float2(0,-4),
	//out ring second 3
	float2(0, 4),
	float2(3, -2),	
	float2(-3, -2),
	//outer outer ring, first 3
	float2(-4, 0),
	float2(2, 4),
	float2(2,-4),
	//outer outer ring, second 3
	float2(-2, 4),
	float2(4, 0),
	float2(-2, -4)
};

/*=============================================================================
	Functions - Common
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
    return o;
}

float3 get_jitter_blue(in int2 pos)
{
	return tex2Dfetch(sBlueNoiseJitterTex, pos % 32).xyz;
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(log(1 + dist * 1000.0  / FRAMETIME )));//normalize by frametime such that we don't need to adjust visualization intensity all the time
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
	OF - Inputs
=============================================================================*/
/*
texture2D StateCounterTex	{ Format = R32F;  	};
sampler2D sStateCounterTex	{ Texture = StateCounterTex;  };

float4 FrameWriteVS(in uint id : SV_VertexID) : SV_Position {return float4(!debug_key_down, !debug_key_down, 0, 1);}
float  FrameWritePS(in float4 vpos : SV_Position) : SV_Target0 {return FRAMECOUNT;}
*/
void WriteCurrFeatureAndDepthPS(in VSOUT i, out float o0 : SV_Target0, out float o1 : SV_Target1)
{	
	o0 = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	o1 = Depth::get_linear_depth(i.uv); 
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x + 1) discard;
}

void WritePrevFeaturePS(in VSOUT i, out float o : SV_Target0)
{	
	o = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);	
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void WritePrevDepthMipPS(in VSOUT i, out float o : SV_Target0)
{
	o = tex2Dlod(sLinearDepthCurr, i.uv, 3).x; //reuse mip calculation for current frame depth	
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void downsample_features(sampler s0, sampler s1, float2 uv, out float2 f0, out float2 f1)
{
	f0 = f1 = 0;
	float wsum = 0;
	float2 tx = rcp(tex2Dsize(s0));		

	[unroll]for(int x = -1; x <= 1; x++)
	[unroll]for(int y = -1; y <= 1; y++)
	{
		float2 offs = float2(x, y) * 2;
		float2 offs_tl = offs + float2(-0.5, -0.5);
		float2 offs_tr = offs + float2( 0.5, -0.5);
		float2 offs_bl = offs + float2(-0.5,  0.5);
		float2 offs_br = offs + float2( 0.5,  0.5);

		float4 g = float4(dot(offs_tl, offs_tl), dot(offs_tr, offs_tr), dot(offs_bl, offs_bl), dot(offs_br, offs_br));
		g = exp(-g * 0.1);
		float tg = dot(g, 1);
		offs = (offs_tl * g.x + offs_tr * g.y + offs_bl * g.z + offs_br * g.w) / tg;
		f0 += tg * tex2Dlod(s0, uv + offs * tx, 0).rg;
		f1 += tg * tex2Dlod(s1, uv + offs * tx, 0).rg;
		wsum += tg;
	}

	f0 /= wsum; f1 /= wsum;
}

void DownsampleFeaturesPS1(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL0, sFlowFeaturesPrevL0, i.uv, f0, f1);} 
void DownsampleFeaturesPS2(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL1, sFlowFeaturesPrevL1, i.uv, f0, f1);} 
void DownsampleFeaturesPS3(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL2, sFlowFeaturesPrevL2, i.uv, f0, f1);} 
void DownsampleFeaturesPS4(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL3, sFlowFeaturesPrevL3, i.uv, f0, f1);} 
void DownsampleFeaturesPS5(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL4, sFlowFeaturesPrevL4, i.uv, f0, f1);} 
void DownsampleFeaturesPS6(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL5, sFlowFeaturesPrevL5, i.uv, f0, f1);}
void DownsampleFeaturesPS7(in VSOUT i, out float2 f0 : SV_Target0, out float2 f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL6, sFlowFeaturesPrevL6, i.uv, f0, f1);}

/*=============================================================================
	OF - OF
=============================================================================*/

struct SophiaOptimizer
{
    float2 m;              
    float  h;              
    float  beta1, beta2;   
    float  epsilon;        
    float  lr;            
    float  rho;
};

// Initialize the Sophia optimizer
SophiaOptimizer init_sophia()
{
    SophiaOptimizer s;
    s.m = 0;
    s.h = 0;
    s.beta1   = 0.965;
    s.beta2   = 0.99;
    s.epsilon = 1e-15;
    s.lr      = 0.000625 * 0.6;  
    s.rho     = 0.04;
    return s;
}

// One update step of Sophia
float2 update_sophia(inout SophiaOptimizer s, float2 grad)
{
    s.m = lerp(grad, s.m, s.beta1);
    float g2 = dot(grad, grad);
    s.h = lerp(g2, s.h, s.beta2);

	//vectorized, replace sign with normalization and abs with length
	float mlen = length(s.m);
	float2 mnorm = s.m / (mlen + s.epsilon);
	
	float ratio = saturate(mlen / (s.rho * s.h + s.epsilon));	
	return s.lr * mnorm * ratio;
}

float4 filter_flow(in VSOUT i, sampler s_flow, const int depth_mip = 3, const int radius = 3)
{	
	float2 txflow = rcp(tex2Dsize(s_flow));
	float depth = tex2Dlod(sLinearDepthCurr, i.uv, depth_mip).x;
	float4 blurred = 0;

	float4 center_flow = tex2Dlod(s_flow, i.uv, 0);

	[loop]for(int y = -radius; y <= radius; y++)
	[loop]for(int x = -radius; x <= radius; x++)
	{		
		float2 tuv = i.uv + txflow * float2(x, y) * 2;
		float4 tap = tex2Dlod(s_flow, tuv, 0);
		float lw = log2(1.0 + max(0, center_flow.z / (tap.z + 1e-6) - 0.5)); //we want to explicitly get better samples from the neighbours
		float zw = exp(-abs(tap.w / (depth + 1e-6) - 1) * 64.0);
		blurred += float4(tap.xyz, 1) * (lw * zw + 1e-7);
	}

	blurred.xyz /= blurred.w;
	return float4(blurred.xyz, tex2Dlod(sLinearDepthPrevLo, i.uv + blurred.xy, 0).x);//write prev frame depth for reprojection validation
}

//can't write to the final flow map when I read it here so
float4 filter_flow_final(in VSOUT i, sampler s_flow, const int depth_mip = 2, const int radius = 3)
{
	float2 txflow = rcp(tex2Dsize(s_flow));
	float depth = tex2Dlod(sLinearDepthCurr, i.uv, depth_mip).x;
	float4 blurred = 0;	

	[loop]for(int y = -radius; y <= radius; y++)
	[loop]for(int x = -radius; x <= radius; x++)
	{
		float2 tuv = i.uv + txflow * float2(x, y) * 2;
		float4 tap = tex2Dlod(s_flow, tuv, 0);
		float lw = exp(-tap.z * 16.0); //regular relative weighting
		float zw = exp(-abs(tap.w / (depth + 1e-6) - 1) * 64.0);
		blurred += float4(tap.xyz, 1) * (lw * zw + 1e-7);
	}

	blurred.xyz /= blurred.w;
	return float4(blurred.xyz, depth);
}

float loss(float a, float b)
{
	float t = a - b;
	return abs(t); //SAD
}

//I calculate gradient of block matching loss, and this requires matching the blocks 3 times for finite differences
//doing tex2Dgather once and manually interpolating makes the algorithm almost twice as fast.
#define FLOW_USE_GATHER_GRADIENT 1

float4 calc_flow(VSOUT i,
					 sampler s_feature_curr, 
					 sampler s_feature_prev, 
					 sampler s_flow, 
					 const int level, 
					 const int blocksize)
{	
	//random block matching kernel rotation helps massively at coarser scales
	float2x2 km;
	{
		float randphi = get_jitter_blue(i.vpos.xy).x;
		float2 sc; sincos(randphi * TAU / 6.0, sc.x, sc.y);
		km = float2x2(sc.y, -sc.x, sc.x, sc.y);		
	}
	
	float2 texelsize = rcp(tex2Dsize(s_feature_curr));
	float2 texsize = tex2Dsize(s_feature_curr);

	float local_block[16];	
	[unroll]for(uint k = 0; k < blocksize; k++) 
	{
		float2 tuv = i.uv + mul(star_kernel[k], km) * texelsize;
#if FLOW_USE_GATHER_GRADIENT					
		float4 texels = tex2DgatherR(s_feature_curr, tuv);
		float2 lambda = frac(tuv * texsize - 0.5);
		local_block[k] = lerp(lerp(texels.w, texels.z, lambda.x), lerp(texels.x, texels.y, lambda.x), lambda.y);		
#else
		local_block[k] = tex2Dlod(s_feature_curr, tuv, 0).x;
#endif		
	}		

	float2 total_motion = 0;
	
	[branch]
	if(level < 7)//if we're not the first pass, do some neighbour pooling to get a better initial guess
	{
		total_motion = filter_flow(i, s_flow, 3, 5).xy;		
	}	
	
	float3 SAD = 0;
	float delta = 0.01;

	[unroll]
	for(uint k = 0; k < blocksize; k++)
	{		
		float2 tuv = i.uv + mul(star_kernel[k], km) * texelsize + total_motion;		
		float3 f;
#if FLOW_USE_GATHER_GRADIENT		
		//I should be interpolating away from the pixel boundaries and flip the gradient to avoid interpolating beyond the pixel bounds
		//but this is nearly as accurate and faster.
		float4 texels = tex2DgatherR(s_feature_prev, tuv);
		float2 lambda = frac(tuv * texsize - 0.5);
		f.x = lerp(lerp(texels.w, texels.z, lambda.x),         lerp(texels.x, texels.y, lambda.x),         lambda.y        );
		f.y = lerp(lerp(texels.w, texels.z, lambda.x + delta), lerp(texels.x, texels.y, lambda.x + delta), lambda.y        );	
		f.z = lerp(lerp(texels.w, texels.z, lambda.x),         lerp(texels.x, texels.y, lambda.x),         lambda.y + delta);
#else 
		f.x = tex2Dlod(s_feature_prev, tuv         						   , 0).x; 
		f.y = tex2Dlod(s_feature_prev, tuv + float2(texelsize.x * delta, 0), 0).x;	
		f.z = tex2Dlod(s_feature_prev, tuv + float2(0, texelsize.y * delta), 0).x;
#endif		
		float g = local_block[k];
		SAD.x += loss(f.x, g);
		SAD.y += loss(f.y, g);
		SAD.z += loss(f.z, g);					
    }

	float2 grad = (SAD.yz - SAD.x) * texsize / delta;

	int num_steps = (4 + level) * (1 + 3 * OPTICAL_FLOW_Q);
	//num_steps = level >= 2 ? 2 * num_steps : num_steps;

	float2 local_motion = 0;
	float2 best_local_motion = 0;
	float  best_SAD = SAD.x;

	SophiaOptimizer sophia = init_sophia();
	sophia.lr *= 1.0 + level;

	[loop]
	for(int j = 0; j < num_steps; j++)
	{
		float2 curr_grad_step = 0;

		if(OPTICAL_FLOW_OPT == 0) //SophiaG
		{
			curr_grad_step = update_sophia(sophia, grad);
		}
		else 
		{
			curr_grad_step = grad / (1e-15 + dot(grad, grad)) * SAD.x;			
		}

		//gradient clipping
		float gg = sqrt(dot(curr_grad_step, curr_grad_step)) + 1e-8;
		float clip_ratio = min(gg, 0.3333 * length(texelsize)) / gg; 
		curr_grad_step *= clip_ratio;	
		
		local_motion -= curr_grad_step;		
		SAD = 0;
	
		[unroll]
		for(uint k = 0; k < blocksize; k++)
		{
			float2 tuv = i.uv + total_motion + local_motion + mul(star_kernel[k], km) * texelsize;			
			float3 f;
#if FLOW_USE_GATHER_GRADIENT
			float4 texels = tex2DgatherR(s_feature_prev, tuv);
			float2 lambda = frac(tuv * texsize - 0.5);
			f.x = lerp(lerp(texels.w, texels.z, lambda.x),         lerp(texels.x, texels.y, lambda.x),         lambda.y        );
			f.y = lerp(lerp(texels.w, texels.z, lambda.x + delta), lerp(texels.x, texels.y, lambda.x + delta), lambda.y        );	
			f.z = lerp(lerp(texels.w, texels.z, lambda.x),         lerp(texels.x, texels.y, lambda.x),         lambda.y + delta);
#else 
			f.x = tex2Dlod(s_feature_prev, tuv,          						 0).x; 
			f.y = tex2Dlod(s_feature_prev, tuv + float2(texelsize.x * delta, 0), 0).x;	
			f.z = tex2Dlod(s_feature_prev, tuv + float2(0, texelsize.y * delta), 0).x;
#endif			
			float g = local_block[k];	
			SAD.x += loss(f.x, g);
			SAD.y += loss(f.y, g);
			SAD.z += loss(f.z, g);			
		}

		grad = (SAD.yz - SAD.x) * texsize / delta;

		[branch]
		if(SAD.x < best_SAD) 
		{
			best_SAD = SAD.x;
			best_local_motion = local_motion;			
		}
		else 
		{
			j++;
		}				
	}

	total_motion += best_local_motion;
	float depth_key = 0;
	[flatten]
	if(level == 0) //upscaling should be bilateral on curr frame depth, more accurate
	{
		depth_key = tex2Dlod(sLinearDepthCurr, i.uv, 2).x; //2 -> upscale
	}
	else //vector pooling makes more sense to measure prev frame reprojection error, less flickery
	{
		depth_key = tex2Dlod(sLinearDepthPrevLo, i.uv + total_motion, 0).x;
	}

	float4 curr_layer = float4(total_motion, best_SAD, depth_key);
	return curr_layer;
}

void FilterFlowPS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexNewB);}
void BlockMatchingPassNewPS7(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL7, sFlowFeaturesPrevL7, sMotionTexNewA, 7, 10);}
void BlockMatchingPassNewPS6(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL6, sFlowFeaturesPrevL6, sMotionTexNewA, 6, 10);}
void BlockMatchingPassNewPS5(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL5, sFlowFeaturesPrevL5, sMotionTexNewA, 5, 10);}
void BlockMatchingPassNewPS4(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL4, sFlowFeaturesPrevL4, sMotionTexNewA, 4, 10);}
void BlockMatchingPassNewPS3(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL3, sFlowFeaturesPrevL3, sMotionTexNewA, 3, 10);}
void BlockMatchingPassNewPS2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL2, sFlowFeaturesPrevL2, sMotionTexNewA, 2, 13);}
void BlockMatchingPassNewPS1(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL1, sFlowFeaturesPrevL1, sMotionTexNewA, 1, 16);}
void BlockMatchingPassNewPS0(in VSOUT i, out float4 o : SV_Target0){o = calc_flow(i, sFlowFeaturesCurrL0, sFlowFeaturesPrevL0, sMotionTexNewA, 0, 16);}

void UpscaleFilter8to4PS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow_final(i, sMotionTexNewB, 2, 3);}
void UpscaleFilter4to2PS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow_final(i, sMotionTexUpscale, 1, 2);}
void UpscaleFilter2to1PS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow_final(i, sMotionTexUpscale2, 0, 1);}

/*=============================================================================
	Shader Entry Points - Normals
=============================================================================*/

void NormalsPS(in VSOUT i, out float4 o : SV_Target0)
{
	const float2 dirs[9] = 
	{
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,-1),//TL
		BUFFER_PIXEL_SIZE_DLSS * float2(0,-1),//T
		BUFFER_PIXEL_SIZE_DLSS * float2(1,-1),//TR
		BUFFER_PIXEL_SIZE_DLSS * float2(1,0),//R
		BUFFER_PIXEL_SIZE_DLSS * float2(1,1),//BR
		BUFFER_PIXEL_SIZE_DLSS * float2(0,1),//B
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,1),//BL
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,0),//L
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,-1)//TL first duplicated at end cuz it might be best pair	
	};

	float z_center = Depth::get_linear_depth(i.uv);
	float3 center_pos = Camera::uv_to_proj(i.uv, Camera::depth_to_z(z_center));

	//z close/far
	float2 z_prev;
	z_prev.x = Depth::get_linear_depth(i.uv + dirs[0]);
	z_prev.y = Depth::get_linear_depth(i.uv + dirs[0] * 2);
	float3 dv_prev = Camera::uv_to_proj(i.uv + dirs[0], Camera::depth_to_z(z_prev.x)) - center_pos;

	float4 best_normal = float4(0,0,0,100000);
	float4 weighted_normal = 0;

	[unroll]
	for(int j = 1; j < 9; j++)
	{
		float2 z_curr;
		z_curr.x = Depth::get_linear_depth(i.uv + dirs[j]);
		z_curr.y = Depth::get_linear_depth(i.uv + dirs[j] * 2);

		float3 dv_curr = Camera::uv_to_proj(i.uv + dirs[j], Camera::depth_to_z(z_curr.x)) - center_pos;	
		float3 temp_normal = cross(dv_curr, dv_prev);

		float2 z_guessed = 2 * float2(z_prev.x, z_curr.x) - float2(z_prev.y, z_curr.y);
		float error = dot(1, abs(z_guessed - z_center));
		
		float w = rcp(dot(temp_normal, temp_normal));
		w *= rcp(error * error + exp2(-32.0));
		
		weighted_normal += float4(temp_normal, 1) * w;	
		best_normal = error < best_normal.w ? float4(temp_normal, error) : best_normal;

		z_prev = z_curr;
		dv_prev = dv_curr;
	}

	float3 normal = weighted_normal.w < 1.0 ? best_normal.xyz : weighted_normal.xyz;
	//normal = best_normal.xyz;
	normal *= rsqrt(dot(normal, normal) + 1e-8);
	//V2 geom normals to .zw	
	o = Math::octahedral_enc(-normal).xyxy;//fixes bugs in RTGI, normal.z positive gives smaller error :)	
}

//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex0  { Width = BUFFER_WIDTH_DLSS/2;   Height = BUFFER_HEIGHT_DLSS/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex0 { Texture = SmoothNormalsTempTex0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; };
//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex1  { Width = BUFFER_WIDTH_DLSS/2;   Height = BUFFER_HEIGHT_DLSS/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex1 { Texture = SmoothNormalsTempTex1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;  };
//high res copy back so we can fetch center tap at full res always
texture SmoothNormalsTempTex2  < pooled = true; > { Width = BUFFER_WIDTH_DLSS;   Height = BUFFER_HEIGHT_DLSS;   Format = RGBA16;  };
sampler sSmoothNormalsTempTex2 { Texture = SmoothNormalsTempTex2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;  };

void SmoothNormalsMakeGbufPS(in VSOUT i, out float4 o : SV_Target0)
{
	o.xyz = Deferred::get_normals(i.uv);
	o.w = Camera::depth_to_z(Depth::get_linear_depth(i.uv));
}

void get_gbuffer(in sampler s, in float2 uv, out float3 p, out float3 n)
{
	float4 t = tex2Dlod(s, uv, 0);
	n = t.xyz;
	p = Camera::uv_to_proj(uv, t.w);
}

void get_gbuffer_hi(in float2 uv, out float3 p, out float3 n)
{
	n = Deferred::get_normals(uv);
	p = Camera::uv_to_proj(uv);
}

float sample_distribution(float x, int iteration)
{
	if(!iteration) return x * sqrt(x);
	return x;
	//return x * x;
	//return exp2(2 * x - 2);
}

float sample_pdf(float x, int iteration)
{
	if(!iteration) return 1.5 * sqrt(x);
	return 1;
	//return 2 * x;
	//return 2 * log(2.0) * exp2(2 * x - 2);
}

float2x3 to_tangent(float3 n)
{
    bool bestside = n.z < n.y;
    float3 n2 = bestside ? n.xzy : n;
    float3 k = (-n2.xxy * n2.xyy) * rcp(1.0 + n2.z) + float3(1, 0, 1);
    float3 u = float3(k.xy, -n2.x);
    float3 v = float3(k.yz, -n2.y);
    u = bestside ? u.xzy : u;
    v = bestside ? v.xzy : v;
    return float2x3(u, v);
}

float4 smooth_normals_mkii(in VSOUT i, int iteration, sampler sGbuffer)
{
	int num_dirs = iteration ? 6 : 4;
	int num_steps = iteration ? 3 : 6;	
	float radius_mult = iteration ? 0.2 : 1.0;	

	float2 angle_tolerance = float2(45.0, 30.0); //min/max

	radius_mult *= 0.2 * 0.2;

	float4 rotator = Math::get_rotator(TAU / num_dirs);
	float2 kernel_dir; sincos(TAU / num_dirs + TAU / 12.0, kernel_dir.x, kernel_dir.y); 
	
	float3 p, n;
	get_gbuffer_hi(i.uv, p, n);
	float2x3 kernel_matrix = to_tangent(n);

	float4 bin_front = float4(n, 1) * 0.001;
	float4 bin_back = float4(n, 1) * 0.001;

	float2 sigma_n = cos(radians(angle_tolerance));

	[loop]
	for(int dir = 0; dir < num_dirs; dir++)
	{
		[loop]
		for(int stp = 0; stp < num_steps; stp++)
		{
			float fi = float(stp + 1.0) / num_steps;

			float r = sample_distribution(fi, iteration);
			float ipdf = sample_pdf(fi, iteration);

			float2 sample_dir = normalize(Camera::proj_to_uv(p + 0.1 * mul(kernel_dir, kernel_matrix)) - i.uv);
			//sample_dir = 0.8 * BUFFER_ASPECT_RATIO * kernel_dir;//
			//sample_dir = kernel_dir * 0.2;

			float2 sample_uv = i.uv + sample_dir * r * radius_mult;
			if(!Math::inside_screen(sample_uv)) break;

			float3 sp, sn;
			get_gbuffer(sGbuffer, sample_uv, sp, sn);

			float ndotn = dot(sn, n);
			float plane_distance = abs(dot(sp - p, n)) + abs(dot(p - sp, sn));

			float wn = smoothstep(sigma_n.x, sigma_n.y, ndotn);
			float wz = exp2(-plane_distance*plane_distance * 10.0);
			float wd = exp2(-dot(p - sp, p - sp));

			float w = wn * wz * wd;

			//focal point detection, find closest point to both 3D lines
			/*
			//find connecting axis
			float3 A = cross(n, sn);

			//find segment lengths for both line equations p + lambda * n
			float d2 = dot(p - sp, cross(n, A)) / dot(sn, cross(n, A));
			float d1 = dot(sp - p, cross(sn, A)) / dot(n, cross(sn, A));
			*/

			//heavily simplified math of the above using Lagrange identity and dot(n,n)==dot(sn,sn)==1
			float d2 = (ndotn * dot(p - sp,  n) - dot(p - sp, sn)) / (ndotn*ndotn - 1);
			float d1 = (ndotn * dot(p - sp, sn) - dot(p - sp,  n)) / (1 - ndotn*ndotn);

			//calculate points where each line is closest to the other line
			float3 hit1 = p + n * d1;
			float3 hit2 = sp + sn * d2;

			//mutual focal point is the mid point between those 2
			float3 middle = (hit1 + hit2) * 0.5;
			float side = dot(middle - p, n);

			//a hard sign split causes flickering, so do a smooth classifier as front or back
			float front_weight = saturate(side * 3.0 + 0.5);
			float back_weight = 1 - front_weight;

			if(ndotn > 0.9999) //fix edge case with parallel lines
			{
				front_weight = 1;
				back_weight = 1;
			}

			bin_front += float4(sn, 1) * ipdf * w * front_weight;
			bin_back += float4(sn, 1) * ipdf * w * back_weight;

			if(w < 0.01) break;
		}

		kernel_dir = Math::rotate_2D(kernel_dir, rotator);
	}

	bin_back.xyz = normalize(bin_back.xyz);
	bin_front.xyz = normalize(bin_front.xyz);

	//smooth binary select
	float bal = bin_back.w / (bin_front.w + bin_back.w);
	bal = smoothstep(0, 1, bal);
	bal = smoothstep(0, 1, bal);

	float3 best_bin = lerp(bin_front.xyz, bin_back.xyz, bal);
	return float4(safenormalize(best_bin), p.z);
}

VSOUT SmoothNormalsVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
	if(!ENABLE_SMOOTH_NORMALS) o.vpos = -100000; //forcing NaN here kills this in geometry stage, faster than discard()
    return o;
}

void SmoothNormalsPass0PS(in VSOUT i, out float4 o : SV_Target0)
{
	o = smooth_normals_mkii(i, 0, sSmoothNormalsTempTex0);	
}

void SmoothNormalsPass1PS(in VSOUT i, out float4 o : SV_Target0)
{	
	float3 n = -smooth_normals_mkii(i, 1, sSmoothNormalsTempTex1).xyz;
	float3 orig_n = n;

	[branch]
	if(ENABLE_TEXTURED_NORMALS)
	{
		float3 p = Camera::uv_to_proj(i.uv);
		float luma = dot(tex2D(ColorInput, i.uv).rgb, 0.3333);

		float3 e_y = (p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE_DLSS * float2(0, 2)));
		float3 e_x = (p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE_DLSS * float2(2, 0)));
		e_y = normalize(cross(n, e_y));
		e_x = normalize(cross(n, e_x));

		float radius_scale = (0.5 + RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.01 * saturate(TEXTURED_NORMALS_RADIUS)) / 50.0;

		float3 v_y = e_y * radius_scale;
		float3 v_x = e_x * radius_scale;

		float3 center_color = tex2D(ColorInput, i.uv).rgb;
		float center_luma = dot(center_color * center_color, float3(0.2126, 0.7152, 0.0722));

		float3 center_p_height = p + center_luma * n;
		float3 summed_normal = n * 0.01;

		int octaves = TEXTURED_NORMALS_QUALITY;	

		float total_luma = center_luma;

		[loop]
		for(int octave = 0; octave < octaves; octave++)
		{
			float3 height[4];
			float4 plane_dist;

			float2 axis; sincos(HALF_PI * octave / float(octaves), axis.y, axis.x); //modulate directions per octave to get better rotation invariance
			const float4 next_axis = Math::get_rotator(HALF_PI);

			float fi = exp2(octave);
			axis *= fi;

			[unroll]
			for(int a = 0; a < 4; a++)
			{
				float3 virtual_p = p + v_x * axis.x + v_y * axis.y;
				float2 uv = Camera::proj_to_uv(virtual_p);	
				float3 actual_p = Camera::uv_to_proj(uv);

				float3 tap_color = tex2Dlod(ColorInput, uv, 0).rgb;
				float tap_luma = dot(tap_color * tap_color, float3(0.2126, 0.7152, 0.0722));
				total_luma += tap_luma;
				
				height[a] = virtual_p + tap_luma * n;
				plane_dist[a] = abs(dot(n, actual_p - p));

				axis = Math::rotate_2D(axis, next_axis);
			}

			[unroll]
			for(int j = 0; j < 4; j++)
			{
				uint this_idx = j;
				uint next_idx = (j + 1) % 4;

				float w = rcp(0.05 + plane_dist[this_idx] + plane_dist[next_idx]);
				float3 curr_n = -cross(height[this_idx] - center_p_height, height[next_idx] - center_p_height);
				curr_n *= rsqrt(1e-5 + dot(curr_n, curr_n));
				w *= exp2(-octave);
				summed_normal += curr_n * w;
			}
		}

		summed_normal.xyz = safenormalize(summed_normal.xyz);
		float3 halfvec = n - summed_normal.xyz * 0.95;
		halfvec.xyz /= lerp(total_luma, 0.5,  0.5);
		n += halfvec * saturate(TEXTURED_NORMALS_INTENSITY * TEXTURED_NORMALS_INTENSITY * TEXTURED_NORMALS_INTENSITY) * 10.0;
		n = normalize(n);
	}	

	o.xy = Math::octahedral_enc(n);
	o.zw = Math::octahedral_enc(orig_n);
}

void CopyNormalsPS(in VSOUT i, out float4 o : SV_Target0)
{
	o = tex2D(sSmoothNormalsTempTex2, i.uv);
}

/*=============================================================================
	Fake albedo texture
=============================================================================*/

float3 srgb_to_AgX(float3 srgb)
{
    float3x3 toagx = float3x3(0.842479, 0.0784336, 0.0792237, 
                              0.042328, 0.8784686, 0.0791661, 
                              0.042376, 0.0784336, 0.8791430);
    return mul(toagx, srgb);         
}

float3 AgX_to_srgb(float3 AgX)
{   
    float3x3 fromagx = float3x3(1.19688,  -0.0980209, -0.0990297,
                               -0.0528969, 1.1519,    -0.0989612,
                               -0.0529716, -0.0980435, 1.15107);
    return mul(fromagx, AgX);            
}

#define degamma(_v) ((_v)*0.283799*((2.52405+(_v))*(_v)))
#define regamma(_v) (1.14374*(-0.126893*(_v)+sqrt(_v)))

float3 unpack_hdr_rtgi(float3 color)
{
    color  = saturate(color);   
    color = degamma(color);
    color = srgb_to_AgX(color);
    color = color * rcp(1.04 - saturate(color));    
    return color;
}

float3 pack_hdr_rtgi(float3 color)
{
    color =  1.04 * color * rcp(color + 1.0);   
    color = AgX_to_srgb(color);    
    color  = saturate(color);
    color = regamma(color);
    return color;     
}

float3 cone_overlap(float3 c)
{
    float k = 0.4 * 0.33;
    float2 f = float2(1 - 2 * k, k);
    float3x3 m = float3x3(f.xyy, f.yxy, f.yyx);
    return mul(c, m);
}

float3 cone_overlap_inv(float3 c)
{
    float k = 0.4 * 0.33;
    float2 f = float2(k - 1, k) * rcp(3 * k - 1);
    float3x3 m = float3x3(f.xyy, f.yxy, f.yyx);
    return mul(c, m);
}

float3 sdr_to_hdr(float3 c)
{ 
	return unpack_hdr_rtgi(c);    
}

float3 hdr_to_sdr(float3 c)
{    
	return pack_hdr_rtgi(c);    
}

float get_sdr_luma(float3 c)
{
    c = degamma(c);
    float lum = dot(c, float3(0.2125, 0.7154, 0.0721));
    lum = regamma(lum);
    return lum;
}

#define ALBEDO_EXPOSURE_TARGET 		0.3
#define ALBEDO_RES_SCALE  			3

#define TILE_WIDTH     (BUFFER_WIDTH / ALBEDO_RES_SCALE)
#define TILE_HEIGHT    (BUFFER_HEIGHT / ALBEDO_RES_SCALE)

//this is really awkward but we cannot use any of the common preprocessor integer log2 macros
//as the preprocessor runs out of stack space with them. So we have to do it manually like this
#if TILE_HEIGHT 	< 128
    #define LOWEST_MIP  6
#elif TILE_HEIGHT < 256
    #define LOWEST_MIP  7
#elif TILE_HEIGHT < 512
    #define LOWEST_MIP  8
#elif TILE_HEIGHT < 1024
    #define LOWEST_MIP  9
#elif TILE_HEIGHT < 2048
    #define LOWEST_MIP  10
#else 
    #error "Unsupported resolution"
#endif

#define TARGET_MIP        ((LOWEST_MIP) - 3)
#define TARGET_MIP_SCALE  (1 << (TARGET_MIP))

#define ATLAS_TILES_X   2
#define ATLAS_TILES_Y   2

//rounded up tile resolution such that it can be cleanly divided by 2 TARGET_MIP'th times
#define ATLAS_TILE_RES_X  CEIL_DIV(TILE_WIDTH, TARGET_MIP_SCALE) * TARGET_MIP_SCALE
#define ATLAS_TILE_RES_Y  CEIL_DIV(TILE_HEIGHT, TARGET_MIP_SCALE) * TARGET_MIP_SCALE
#define ATLAS_RES_X ((ATLAS_TILE_RES_X) * (ATLAS_TILES_X))
#define ATLAS_RES_Y ((ATLAS_TILE_RES_Y) * (ATLAS_TILES_Y))

texture LPWeightAtlasL0    	{ Width = (ATLAS_RES_X)>>0; Height = (ATLAS_RES_Y)>>0; Format = RGBA16F;};
sampler sLPWeightAtlasL0   	{ Texture = LPWeightAtlasL0;};
#if TARGET_MIP >= 1
texture LPWeightAtlasL1Tmp  { Width = (ATLAS_RES_X)>>1; Height = (ATLAS_RES_Y)>>0; Format = RGBA16F;};
sampler sLPWeightAtlasL1Tmp { Texture = LPWeightAtlasL1Tmp;};
texture LPWeightAtlasL1    	{ Width = (ATLAS_RES_X)>>1; Height = (ATLAS_RES_Y)>>1; Format = RGBA16F;};
sampler sLPWeightAtlasL1   	{ Texture = LPWeightAtlasL1;};
#endif
#if TARGET_MIP >= 2
texture LPWeightAtlasL2Tmp  { Width = (ATLAS_RES_X)>>2; Height = (ATLAS_RES_Y)>>1; Format = RGBA16F;};
sampler sLPWeightAtlasL2Tmp { Texture = LPWeightAtlasL2Tmp;};
texture LPWeightAtlasL2    	{ Width = (ATLAS_RES_X)>>2; Height = (ATLAS_RES_Y)>>2; Format = RGBA16F;};
sampler sLPWeightAtlasL2   	{ Texture = LPWeightAtlasL2;};
#endif
#if TARGET_MIP >= 3
texture LPWeightAtlasL3Tmp  { Width = (ATLAS_RES_X)>>3; Height = (ATLAS_RES_Y)>>2; Format = RGBA16F;};
sampler sLPWeightAtlasL3Tmp { Texture = LPWeightAtlasL3Tmp;};
texture LPWeightAtlasL3    	{ Width = (ATLAS_RES_X)>>3; Height = (ATLAS_RES_Y)>>3; Format = RGBA16F;};
sampler sLPWeightAtlasL3   	{ Texture = LPWeightAtlasL3;};
#endif
#if TARGET_MIP >= 4
texture LPWeightAtlasL4Tmp  { Width = (ATLAS_RES_X)>>4; Height = (ATLAS_RES_Y)>>3; Format = RGBA16F;};
sampler sLPWeightAtlasL4Tmp { Texture = LPWeightAtlasL4Tmp;};
texture LPWeightAtlasL4    	{ Width = (ATLAS_RES_X)>>4; Height = (ATLAS_RES_Y)>>4; Format = RGBA16F;};
sampler sLPWeightAtlasL4   	{ Texture = LPWeightAtlasL4;};
#endif
#if TARGET_MIP >= 5
texture LPWeightAtlasL5Tmp  { Width = (ATLAS_RES_X)>>5; Height = (ATLAS_RES_Y)>>4; Format = RGBA16F;};
sampler sLPWeightAtlasL5Tmp { Texture = LPWeightAtlasL5Tmp;};
texture LPWeightAtlasL5    	{ Width = (ATLAS_RES_X)>>5; Height = (ATLAS_RES_Y)>>5; Format = RGBA16F;};
sampler sLPWeightAtlasL5   	{ Texture = LPWeightAtlasL5;};
#endif
#if TARGET_MIP >= 6
texture LPWeightAtlasL6Tmp  { Width = (ATLAS_RES_X)>>6; Height = (ATLAS_RES_Y)>>5; Format = RGBA16F;};
sampler sLPWeightAtlasL6Tmp { Texture = LPWeightAtlasL6Tmp;};
texture LPWeightAtlasL6    	{ Width = (ATLAS_RES_X)>>6; Height = (ATLAS_RES_Y)>>6; Format = RGBA16F;};
sampler sLPWeightAtlasL6   	{ Texture = LPWeightAtlasL6;};
#endif
#if TARGET_MIP >= 7
texture LPWeightAtlasL7Tmp  { Width = (ATLAS_RES_X)>>7; Height = (ATLAS_RES_Y)>>6; Format = RGBA16F;};
sampler sLPWeightAtlasL7Tmp { Texture = LPWeightAtlasL7Tmp;};
texture LPWeightAtlasL7    	{ Width = (ATLAS_RES_X)>>7; Height = (ATLAS_RES_Y)>>7; Format = RGBA16F;};
sampler sLPWeightAtlasL7   	{ Texture = LPWeightAtlasL7;};
#endif
texture LPFusedExposureTex 	{ Width = ATLAS_TILE_RES_X; Height = ATLAS_TILE_RES_Y; Format = RG16F;};
sampler sLPFusedExposureTex { Texture = LPFusedExposureTex;};

float4 channel_indices_to_bias(int4 channels)
{
    float exposure_spread = 1.0;
    float num_channels = ATLAS_TILES_X * ATLAS_TILES_Y * 4;
    float4 exposure_bias = (float4(channels) - (num_channels - 1) * 0.5) * exposure_spread;
    return exposure_bias;
}

void InitAtlasPyramidPS(in VSOUT i, out float4 o : SV_Target0)
{
    float2 num_tiles = float2(ATLAS_TILES_X, ATLAS_TILES_Y);

    int2 tile_id = floor(i.uv * num_tiles);
    float2 tile_uv = frac(i.uv * num_tiles);

    int4 curr_channel_idx = (tile_id.y * num_tiles.x + tile_id.x) * 4 + int4(0, 1, 2, 3); //x4 -> channels
    float4 exposure_bias = channel_indices_to_bias(curr_channel_idx); 
    exposure_bias = exp2(exposure_bias);  

	//it doesn't fit exactly, since our target texture is a bit larger, but good enough
    float3 c = 0;
    [unroll]for(int x = 0; x < ALBEDO_RES_SCALE; x++)
    [unroll]for(int y = 0; y < ALBEDO_RES_SCALE; y++)
    {
        float2 offs = float2(x, y) - ALBEDO_RES_SCALE * 0.5;
        c += sdr_to_hdr(tex2Dlod(ColorInput, tile_uv + offs * BUFFER_PIXEL_SIZE, 0).rgb);
    }
    c /= ALBEDO_RES_SCALE * ALBEDO_RES_SCALE;  

    float3 exposed[4];
    exposed[0] = hdr_to_sdr(c * exposure_bias.x);
    exposed[1] = hdr_to_sdr(c * exposure_bias.y);
    exposed[2] = hdr_to_sdr(c * exposure_bias.z); 
    exposed[3] = hdr_to_sdr(c * exposure_bias.w);  

    float4 luminances;
    luminances.x = get_sdr_luma(exposed[0]);
    luminances.y = get_sdr_luma(exposed[1]);
    luminances.z = get_sdr_luma(exposed[2]);
    luminances.w = get_sdr_luma(exposed[3]); 

    o = exp(-(luminances-0.5)*(luminances-0.5)*32.0); 
}

float4 tile_downsample_separable(const sampler s0, float2 uv, const bool horizontal)
{
    const float offs[12] = {-10.9242,  -8.93782, -6.95154, -4.96533, -2.97918, -0.993056, 0.993056, 2.97918, 4.96533, 6.95154,  8.93782,  10.9242};
    const float wght[12] = { 0.375591, 0.652116, 1.01393 ,   1.4118,   1.7604,   1.96577,  1.96577,  1.7604,  1.4118, 1.01393, 0.652116, 0.375591};

    const float2 num_tiles = float2(ATLAS_TILES_X, ATLAS_TILES_Y);
    const float2 tile_uv_size = rcp(num_tiles);
    float2 tile_mid_uv = (floor(uv * num_tiles) + 0.5) * tile_uv_size;   
    const float2 texelsize = rcp(tex2Dsize(s0, 0));

    float4 blurred_weights = 0;
    float ws = 1e-7;    
    float2 axis = horizontal ? float2(texelsize.x, 0) : float2(0, texelsize.y);

    [unroll]
    for(int j = 0; j < 12; j++)
    {        
        float2 tap_uv = uv + offs[j] * axis;        
        float w = wght[j];    

        //this compiles out
        if(horizontal) 
            w *= step(abs(tap_uv.x - tile_mid_uv.x), tile_uv_size.x * 0.5); 
        else 
            w *= step(abs(tap_uv.y - tile_mid_uv.y), tile_uv_size.y * 0.5); 
        
        blurred_weights += tex2Dlod(s0, tap_uv, 0) * w;
        ws += w;         
    }

    return blurred_weights / ws;
}

#if TARGET_MIP >= 1
void DownsamplePyramidsPS0H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL0, i.uv, true);}
void DownsamplePyramidsPS0V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL1Tmp, i.uv, false);}
#endif
#if TARGET_MIP >= 2
void DownsamplePyramidsPS1H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL1, i.uv, true);}
void DownsamplePyramidsPS1V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL2Tmp, i.uv, false);}
#endif
#if TARGET_MIP >= 3
void DownsamplePyramidsPS2H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL2, i.uv, true);}
void DownsamplePyramidsPS2V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL3Tmp, i.uv, false);}
#endif
#if TARGET_MIP >= 4
void DownsamplePyramidsPS3H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL3, i.uv, true);}
void DownsamplePyramidsPS3V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL4Tmp, i.uv, false);}
#endif
#if TARGET_MIP >= 5
void DownsamplePyramidsPS4H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL4, i.uv, true);}
void DownsamplePyramidsPS4V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL5Tmp, i.uv, false);}
#endif
#if TARGET_MIP >= 6
void DownsamplePyramidsPS5H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL5, i.uv, true);}
void DownsamplePyramidsPS5V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL6Tmp, i.uv, false);}
#endif
#if TARGET_MIP >= 7
void DownsamplePyramidsPS6H(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL6, i.uv, true);}
void DownsamplePyramidsPS6V(in VSOUT i, out float4 o : SV_Target0){o = tile_downsample_separable(sLPWeightAtlasL7Tmp, i.uv, false);}
#endif

void fetch_layers(sampler s, float2 uv, out float4 layers[4])
{
    const float2 num_tiles = float2(ATLAS_TILES_X, ATLAS_TILES_Y);
    float2 tile_res = int2(tex2Dsize(s, 0)) / int2(num_tiles);
    float2 texelsize = rcp(tile_res);

    uv = clamp(uv, texelsize * 0.5, 1 - texelsize * 0.5);

    [unroll]
    for(int j = 0; j < ATLAS_TILES_X * ATLAS_TILES_Y; j++)
    {
        int x = j % ATLAS_TILES_X;
        int y = j / ATLAS_TILES_X;
        float2 tile_uv = (uv + float2(x, y)) / num_tiles; 
        layers[j] = tex2Dlod(s, tile_uv, 0);
    }
}

float hadamard(float4 A[4], float4 B[4])
{
    return dot(A[0], B[0]) + dot(A[1], B[1]) + dot(A[2], B[2]) + dot(A[3], B[3]);
}

float weightnorm(float4 v[4])
{
    float sum = (v[0].x + v[0].y) + (v[0].z + v[0].w);
    sum      += (v[1].x + v[1].y) + (v[1].z + v[1].w);
    sum      += (v[2].x + v[2].y) + (v[2].z + v[2].w);
    sum      += (v[3].x + v[3].y) + (v[3].z + v[3].w);
    return rcp(sum + 1e-6);
}

float balance(int layer)
{
    float x = float(layer)/float(TARGET_MIP);
    return exp2(-x * 6.0);
}

void FuseExposuresPS(in VSOUT i, out float2 o : SV_Target0)
{
    float collapsed = 0;

    float4 G[4];
	G[0] = channel_indices_to_bias(int4(0,  1,  2,  3));
    G[1] = channel_indices_to_bias(int4(4,  5,  6,  7));
    G[2] = channel_indices_to_bias(int4(8,  9,  10, 11));
    G[3] = channel_indices_to_bias(int4(12, 13, 14, 15));
    float4 W[4];
    float wnorm;
  
    fetch_layers(sLPWeightAtlasL0, i.uv, W);
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(0); 
#if TARGET_MIP >= 1 
    collapsed -= hadamard(G, W) * wnorm * balance(1);  
    fetch_layers(sLPWeightAtlasL1, i.uv, W); 
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(1); 
#endif
#if TARGET_MIP >= 2
    collapsed -= hadamard(G, W) * wnorm * balance(2); 
    fetch_layers(sLPWeightAtlasL2, i.uv, W); 
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(2); 
#endif 
#if TARGET_MIP >= 3
    collapsed -= hadamard(G, W) * wnorm * balance(3);
    fetch_layers(sLPWeightAtlasL3, i.uv, W);
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(3);
#endif 
#if TARGET_MIP >= 4
    collapsed -= hadamard(G, W) * wnorm * balance(4); 
    fetch_layers(sLPWeightAtlasL4, i.uv, W); 
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(4); 
#endif 
#if TARGET_MIP >= 5
    collapsed -= hadamard(G, W) * wnorm * balance(5); 
    fetch_layers(sLPWeightAtlasL5, i.uv, W);
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(5); 
 #endif 
#if TARGET_MIP >= 6
    collapsed -= hadamard(G, W) * wnorm * balance(6);
    fetch_layers(sLPWeightAtlasL6, i.uv, W);
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(6);
#endif 
#if TARGET_MIP >= 7
    collapsed -= hadamard(G, W) * wnorm * balance(7); 
    fetch_layers(sLPWeightAtlasL7, i.uv, W);
    wnorm = weightnorm(W);
    collapsed += hadamard(G, W) * wnorm * balance(7);
#endif 
    o.x = collapsed; //collapsed pyramid
    //o.y = get_sdr_luma(tex2D(ColorInput, i.uv).rgb);
	//correctly blurred luma for the guided upsampling
    o.y = 0;
    [unroll]for(int x = 0; x < ALBEDO_RES_SCALE; x++)
    [unroll]for(int y = 0; y < ALBEDO_RES_SCALE; y++)
    {
        float2 offs = float2(x, y) - ALBEDO_RES_SCALE * 0.5;
        o.y += get_sdr_luma(tex2Dlod(ColorInput, i.uv + offs * BUFFER_PIXEL_SIZE, 0).rgb);
    }
    o.y /= ALBEDO_RES_SCALE * ALBEDO_RES_SCALE; 
}

void UpsampleAtlasPS(in VSOUT i, out float3 o : SV_Target0)
{
    float3 c = tex2D(ColorInput, i.uv).rgb;
    
	float luminance = get_sdr_luma(c); 

    float4 m = 0; //guide, guide^2, guide*signal, signal
    float ws = 0.0;

    float2 texelsize = rcp(tex2Dsize(sLPFusedExposureTex, 0));

	float2 minmax = float2(1000, -1000);
    [unroll]for(int x = -1; x <= 1; x++)  
    [unroll]for(int y = -1; y <= 1; y++) 
    {
        float2 offs = float2(x, y);
        float2 t = tex2D(sLPFusedExposureTex, i.uv + offs * texelsize).xy;
        float w = exp(-dot(offs, offs));
        m += float4(t.y, t.y * t.y, t.y * t.x, t.x) * w;
        ws += w;

		minmax.x = min(minmax.x, t.x);
		minmax.y = max(minmax.y, t.x);
    }    

    m /= ws;
    
    float a = (m.z - m.x * m.w) / (max(m.y - m.x * m.x, 0.0) + 0.00001);
    float b = m.w - a * m.x;

    float exposure_ratio = a * luminance + b;
	exposure_ratio = clamp(exposure_ratio, minmax.x, minmax.y);
    exposure_ratio = exp2(exposure_ratio);

    o.rgb = tex2D(ColorInput, i.uv).rgb;
    o.rgb = sdr_to_hdr(o.rgb);
    o.rgb *= exposure_ratio;

    //creates less halos this way as it creates less sharp transitions between exposure brackets if exposure target is low/high
    float3 target_hdr = sdr_to_hdr(ALBEDO_EXPOSURE_TARGET.xxx);
    float3 current_hdr = sdr_to_hdr(0.5);
    o.rgb *= target_hdr / current_hdr;

	//Let L = lighting, A = albedo, C = final scene color, p = multiscatter probability
	//then  C = L * (A + p * A + p² * A ....)
	//this means the final color is a combination of single and multiscattering. I'm fudging things here with a constant light
	//but if we invert this MacLaurin series to get the actual albedo A, we get... a reinhard tonemap curve lmao	
	float3 L = 1; //assumed lighting
	float3 C = o.rgb;
	float p = 1.0; //backscatter probability, lambert we assume 1
	o.rgb = C / (L + C * p);
	return;
}

texture RTGI_AlbedoTexV3      { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler sRTGI_AlbedoTexV3     { Texture = RTGI_AlbedoTexV3; };

/*=============================================================================
	Debug
=============================================================================*/

#if LAUNCHPAD_DEBUG_OUTPUT != 0
void DebugPS(in VSOUT i, out float3 o : SV_Target0)
{	
	o = 0;
	switch(DEBUG_MODE)
	{
		case 0: //all 
		{
			float2 tuv = i.uv * 2.0;
			int2 q = tuv < 1.0.xx ? int2(0,0) : int2(1,1);
			tuv = frac(tuv);
			int qq = q.x * 2 + q.y;
			if(qq == 0) o = Deferred::get_normals(tuv) * 0.5 + 0.5;
			if(qq == 1) o = gradient(Depth::get_linear_depth(tuv));
			//if(qq == 2) o = showmotion(Deferred::get_motion(tuv));	
			if(qq == 3) o = tex2Dlod(ColorInput, tuv, 0).rgb;	
			break;			
		}
		case 1: o = showmotion(Deferred::get_motion(i.uv)); break;
		case 2: o = tex2Dlod(ColorInput, i.uv, 0).rgb; break;		
		case 3: o = Deferred::get_normals(i.uv) * 0.5 + 0.5; break;
		case 4: o = gradient(Depth::get_linear_depth(i.uv)); break;
	}
}

#define FLOW_VECTOR_DENSITY_INV 16
#define NUM_VECTORS_X (BUFFER_WIDTH / FLOW_VECTOR_DENSITY_INV)
#define NUM_VECTORS_Y (BUFFER_HEIGHT / FLOW_VECTOR_DENSITY_INV / 0.866)

void FlowVectorDebugVS(in uint id : SV_VertexID, out float4 vpos : SV_Position, out float2 uv : TEXCOORD0, out float4 color : LINECOLOR)
{
	if(DEBUG_MODE != 2) 
	{
		vpos = color = -100000;
		return;
	}

	uint tri_id = id / 3;

	float2 gridpos = float2(tri_id % NUM_VECTORS_X, tri_id / NUM_VECTORS_X);
	gridpos = (gridpos + 0.5) / float2(NUM_VECTORS_X, NUM_VECTORS_Y);
	gridpos.x += 0.5 / NUM_VECTORS_X * ((tri_id / NUM_VECTORS_X) % 2);

	float2 mv = Deferred::get_motion(gridpos) * BUFFER_SCREEN_SIZE.xy;	
	float s = length(mv);
	float2 d = mv / (1e-8 + s);	

	float2x2 shape_mat = float2x2(d.x, -d.y, d.y, d.x);
	float2x2 scale_mat = float2x2(s, 0, 0, 4.0);

	shape_mat = mul(shape_mat, scale_mat);

	const float2 tri_offsets[3] = 
	{
		float2(1, 0),
		float2(-0.5, 0.866),
		float2(-0.5, -0.866)
	};
	
	uv = tri_offsets[id % 3];
	vpos.xy = gridpos + mul(shape_mat, uv) * BUFFER_PIXEL_SIZE.xy;
	vpos  = float4(vpos.xy * float2(2, -2) + float2(-1, 1), 0, 1);
	color = float4(showmotion(mv), 1);
}

float4 FlowVectorDebugPS(in float4 vpos : SV_Position, in float2 uv : TEXCOORD0, in float4 color : LINECOLOR) : SV_Target0
{
	float r = length(uv);
	color.w *= smoothstep(0.5, 0.5-fwidth(r), r);	
	return color;
}

#endif

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_Launchpad
<
    ui_label = "iMMERSE: Launchpad (enable and move to the top!)";
    ui_tooltip =        
        "                           MartysMods - Launchpad                             \n"
        "                   MartysMods Epic ReShade Effects (iMMERSE)                  \n"
        "______________________________________________________________________________\n"
        "\n"

        "Launchpad is a catch-all setup shader that prepares various data for the other\n"
        "effects. Enable this effect and move it to the top of the effect list.        \n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                            \n"
        "\n"       
        "______________________________________________________________________________";
>
{    
	//pass {PrimitiveTopology = POINTLIST;VertexCount = 1;VertexShader = FrameWriteVS;PixelShader  = FrameWritePS;RenderTarget = StateCounterTex;} 

	//OF
	pass {VertexShader = MainVS;PixelShader = WriteCurrFeatureAndDepthPS;RenderTarget0 = FlowFeaturesCurrL0;RenderTarget1 = LinearDepthCurr;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS1;RenderTarget0 = FlowFeaturesCurrL1;RenderTarget1 = FlowFeaturesPrevL1;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS2;RenderTarget0 = FlowFeaturesCurrL2;RenderTarget1 = FlowFeaturesPrevL2;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS3;RenderTarget0 = FlowFeaturesCurrL3;RenderTarget1 = FlowFeaturesPrevL3;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS4;RenderTarget0 = FlowFeaturesCurrL4;RenderTarget1 = FlowFeaturesPrevL4;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS5;RenderTarget0 = FlowFeaturesCurrL5;RenderTarget1 = FlowFeaturesPrevL5;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS6;RenderTarget0 = FlowFeaturesCurrL6;RenderTarget1 = FlowFeaturesPrevL6;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturesPS7;RenderTarget0 = FlowFeaturesCurrL7;RenderTarget1 = FlowFeaturesPrevL7;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS7;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS6;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS5;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS4;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS3;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS2;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS1;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = FilterFlowPS;	RenderTarget = MotionTexNewA;}		
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassNewPS0;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = UpscaleFilter8to4PS;	RenderTarget = MotionTexUpscale;}
	pass {VertexShader = MainVS;PixelShader = UpscaleFilter4to2PS;	RenderTarget = MotionTexUpscale2;}
	pass {VertexShader = MainVS;PixelShader = UpscaleFilter2to1PS;	RenderTarget = Deferred::MotionVectorsTex;}
	pass {VertexShader = MainVS;PixelShader = WritePrevFeaturePS;RenderTarget0 = FlowFeaturesPrevL0;}
	pass {VertexShader = MainVS;PixelShader = WritePrevDepthMipPS;RenderTarget0 = LinearDepthPrevLo;}

	//Smooth Normals
	pass {VertexShader = MainVS;PixelShader = NormalsPS; RenderTarget = Deferred::NormalsTexV3; }	
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsMakeGbufPS;  RenderTarget = SmoothNormalsTempTex0;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass0PS;  RenderTarget = SmoothNormalsTempTex1;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass1PS;  RenderTarget = SmoothNormalsTempTex2;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = CopyNormalsPS; RenderTarget = Deferred::NormalsTexV3; }

	//RTGI Albedo Map
	pass{VertexShader = MainVS; PixelShader = InitAtlasPyramidPS;   RenderTarget0 = LPWeightAtlasL0;}	
#if TARGET_MIP >= 1
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS0H;  RenderTarget0 = LPWeightAtlasL1Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS0V;  RenderTarget0 = LPWeightAtlasL1; } 
#endif
#if TARGET_MIP >= 2
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS1H;  RenderTarget0 = LPWeightAtlasL2Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS1V;  RenderTarget0 = LPWeightAtlasL2; }
#endif
#if TARGET_MIP >= 3 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS2H;  RenderTarget0 = LPWeightAtlasL3Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS2V;  RenderTarget0 = LPWeightAtlasL3; }
#endif
#if TARGET_MIP >= 4 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS3H;  RenderTarget0 = LPWeightAtlasL4Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS3V;  RenderTarget0 = LPWeightAtlasL4; } 
#endif
#if TARGET_MIP >= 5 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS4H;  RenderTarget0 = LPWeightAtlasL5Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS4V;  RenderTarget0 = LPWeightAtlasL5; }
#endif
#if TARGET_MIP >= 6 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS5H;  RenderTarget0 = LPWeightAtlasL6Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS5V;  RenderTarget0 = LPWeightAtlasL6; }
#endif
#if TARGET_MIP >= 7 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS6H;  RenderTarget0 = LPWeightAtlasL7Tmp; } 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS6V;  RenderTarget0 = LPWeightAtlasL7; }
#endif
    pass{VertexShader = MainVS; PixelShader = FuseExposuresPS;  RenderTarget0 = LPFusedExposureTex;}  
	pass{VertexShader = MainVS; PixelShader = UpsampleAtlasPS;  RenderTarget = RTGI_AlbedoTexV3;}

#if LAUNCHPAD_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }	
	pass 
	{
		PrimitiveTopology = TRIANGLELIST;
		VertexCount = NUM_VECTORS_X * NUM_VECTORS_Y * 3;
		VertexShader = FlowVectorDebugVS;
		PixelShader  = FlowVectorDebugPS;
		BlendEnable=true;
		BlendOp=ADD;
		SrcBlend=SRCALPHA;
		DestBlend=INVSRCALPHA;
	} 		
#endif
}
