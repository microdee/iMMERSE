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

	This contains only the old optical flow code, which was far better for
	motion based visual effects (like motion blur)

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef OPTICAL_FLOW_MATCHING_LAYERS 
 #define OPTICAL_FLOW_MATCHING_LAYERS 	2		//[0-2] 0=luma, 1=luma + depth, 2 = rgb + depth
#endif

#ifndef OPTICAL_FLOW_RESOLUTION
 #define OPTICAL_FLOW_RESOLUTION 		1		//[0-2] 0=fullres, 1=halfres, 2=quarter res
#endif

#ifndef VELOCITY_DEBUG_OUTPUT
 #define VELOCITY_DEBUG_OUTPUT 	  	0		//[0 or 1] 1: enables debug output of the motion vectors
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float FILTER_RADIUS <
	ui_type = "drag";
	ui_label = "Optical Flow Filter Smoothness";
	ui_min = 0.0;
	ui_max = 6.0;
	ui_category = "Optical Flow";		
> = 0.2;

#if VELOCITY_DEBUG_OUTPUT != 0
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
	"OPTICAL_FLOW_MATCHING_LAYERS\n"
	"\n"
	"Determines which data to use for optical flow\n"
	"0: luma (fastest)\n"
	"1: luma + depth (more accurate, slower, recommended)\n"
	"2: circular harmonics (by far most accurate, slowest)\n"
	"\n"
	"OPTICAL_FLOW_RESOLUTION\n"
	"\n"
	"Resolution factor for optical flow\n"
	"0: full resolution (slowest)\n"
	"1: half resolution (faster, recommended)\n"
	"2: quarter resolution (fastest)\n"
	"\n"
	"VELOCITY_DEBUG_OUTPUT\n"
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

#if __RENDERER__ < RENDERER_D3D10 //too many textures because DX9 is a jackass
 #if OPTICAL_FLOW_MATCHING_LAYERS == 2
 #undef OPTICAL_FLOW_MATCHING_LAYERS 
 #define OPTICAL_FLOW_MATCHING_LAYERS 1
 #endif	
#endif

#define INTERP 			LINEAR
#define FILTER_WIDE	 	true 
#define FILTER_NARROW 	false

#define SEARCH_OCTAVES              2
#define OCTAVE_SAMPLES             	4

uniform uint FRAMECOUNT < source = "framecount"; >;

#define MAX_MIP  	6 //do not change, tied to textures
#define MIN_MIP 	OPTICAL_FLOW_RESOLUTION

texture OldMotionTexIntermediate6               { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RGBA16F;  };
sampler sOldMotionTexIntermediate6              { Texture = OldMotionTexIntermediate6; };
texture OldMotionTexIntermediate5               { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RGBA16F;  };
sampler sOldMotionTexIntermediate5              { Texture = OldMotionTexIntermediate5; };
texture OldMotionTexIntermediate4               { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RGBA16F;  };
sampler sOldMotionTexIntermediate4              { Texture = OldMotionTexIntermediate4; };
texture OldMotionTexIntermediate3               { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F;  };
sampler sOldMotionTexIntermediate3              { Texture = OldMotionTexIntermediate3; };
texture OldMotionTexIntermediate2               { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;  };
sampler sOldMotionTexIntermediate2              { Texture = OldMotionTexIntermediate2; };
texture OldMotionTexIntermediate1               { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;  };
sampler sOldMotionTexIntermediate1              { Texture = OldMotionTexIntermediate1; };

#define OldMotionTexIntermediate0 				Velocity::OldMotionVectorsTex
#define sOldMotionTexIntermediate0 			Velocity::sOldMotionVectorsTex

#if OPTICAL_FLOW_MATCHING_LAYERS == 0
 #define FEATURE_FORMAT 	R8 
 #define FEATURE_TYPE 		float
 #define FEATURE_COMPS 		x
#else
 #define FEATURE_FORMAT 	RG16F
 #define FEATURE_TYPE		float2
 #define FEATURE_COMPS 		xy
#endif

texture OldFeatureLayerPyramid          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = FEATURE_FORMAT; MipLevels = 1 + MAX_MIP - MIN_MIP; };
sampler sOldFeatureLayerPyramid         { Texture = OldFeatureLayerPyramid; MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; }; 
texture OldFeatureLayerPyramidPrev          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = FEATURE_FORMAT; MipLevels = 1 + MAX_MIP - MIN_MIP; };
sampler sOldFeatureLayerPyramidPrev         { Texture = OldFeatureLayerPyramidPrev;MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; };

#if OPTICAL_FLOW_MATCHING_LAYERS == 2
texture OldCircularHarmonicsPyramidCurr0          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = RGBA16F; MipLevels = 4 - MIN_MIP; };
sampler sOldCircularHarmonicsPyramidCurr0         { Texture = OldCircularHarmonicsPyramidCurr0; MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; }; 
texture OldCircularHarmonicsPyramidCurr1          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = RGBA16F; MipLevels = 4 - MIN_MIP; };
sampler sOldCircularHarmonicsPyramidCurr1         { Texture = OldCircularHarmonicsPyramidCurr1; MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; }; 
texture OldCircularHarmonicsPyramidPrev0          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = RGBA16F; MipLevels = 4 - MIN_MIP; };
sampler sOldCircularHarmonicsPyramidPrev0         { Texture = OldCircularHarmonicsPyramidPrev0; MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; }; 
texture OldCircularHarmonicsPyramidPrev1          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = RGBA16F; MipLevels = 4 - MIN_MIP; };
sampler sOldCircularHarmonicsPyramidPrev1         { Texture = OldCircularHarmonicsPyramidPrev1; MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; }; 
#else 
 #define OldCircularHarmonicsPyramidCurr0 ColorInputTex
 #define OldCircularHarmonicsPyramidCurr1 ColorInputTex
 #define OldCircularHarmonicsPyramidPrev0 ColorInputTex
 #define OldCircularHarmonicsPyramidPrev1 ColorInputTex
 #define sOldCircularHarmonicsPyramidCurr0 ColorInput
 #define sOldCircularHarmonicsPyramidCurr1 ColorInput
 #define sOldCircularHarmonicsPyramidPrev0 ColorInput
 #define sOldCircularHarmonicsPyramidPrev1 ColorInput
#endif

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

/*=============================================================================
	Functions
=============================================================================*/

float4 get_curr_feature(float2 uv, int mip)
{
	mip = max(0, mip - MIN_MIP);
	return tex2Dlod(sOldFeatureLayerPyramid, saturate(uv), mip);
}

float4 get_prev_feature(float2 uv, int mip)
{
	mip = max(0, mip - MIN_MIP);
	return tex2Dlod(sOldFeatureLayerPyramidPrev, saturate(uv), mip);
}

float get_similarity(FEATURE_TYPE m_xx, FEATURE_TYPE m_yy, FEATURE_TYPE m_xy)
{	
#if OPTICAL_FLOW_MATCHING_LAYERS == 0
	return m_xy * rsqrt(m_xx * m_yy);
#else
	return dot(0.5, m_xy) * rsqrt(dot(0.5, m_xx) * dot(0.5, m_yy));
#endif
}

float3 jitter(in int2 pos)
{    
    const float2 magicdot = float2(0.75487766624669276, 0.569840290998); 
    const float3 magicadd = float3(0, 0.025, 0.0125) * dot(magicdot, 1);
    return frac(dot(pos, magicdot) + magicadd);  
}

float4 block_matching(VSOUT i, int level, float4 coarse_layer, const int blocksize)
{	
	level = max(level - 1, 0); //sample one higher res for better quality
	float2 texelsize = rcp(tex2Dsize(sOldFeatureLayerPyramid, max(0, level - MIN_MIP)));
	
	FEATURE_TYPE local_block[16];

	float2 total_motion = coarse_layer.xy;
	float coarse_sim = coarse_layer.w;

	FEATURE_TYPE m_x = 0;
	FEATURE_TYPE m_xx = 0;
	FEATURE_TYPE m_yy = 0;
	FEATURE_TYPE m_xy = 0;

	float search_scale = 3;
	i.uv -= texelsize * (blocksize / 2) * search_scale; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	[unroll] //array index not natively addressable bla...
	for(uint k = 0; k < blocksize * blocksize; k++)
	{
		float2 offs = float2(k % blocksize, k / blocksize);
		float2 tuv = i.uv + offs * texelsize * search_scale;
		FEATURE_TYPE t_local = get_curr_feature(tuv, level).FEATURE_COMPS; 	
		FEATURE_TYPE t_search = get_prev_feature(tuv + total_motion, level).FEATURE_COMPS;		

		local_block[k] = t_local;

		m_x += t_local;
		m_xx += t_local * t_local;
		m_yy += t_search * t_search;
		m_xy += t_local * t_search;

	}

	float variance = abs(m_xx.x / (blocksize * blocksize) - m_x.x * m_x.x / ((blocksize * blocksize)*(blocksize * blocksize)));
	float best_sim = minc(m_xy * rsqrt(m_xx * m_yy));

	//this fixes completely white areas from polluting the buffer with false offsets
	if(variance < exp(-32.0) || best_sim > 0.999999) 
		return float4(coarse_layer.xy, 0, 0);

	float phi = radians(360.0 / OCTAVE_SAMPLES);
	float4 rotator = Math::get_rotator(phi);	
	float randseed = jitter(i.vpos.xy).x;
	randseed = QMC::roberts1(level, randseed);

	float2 randdir; sincos(randseed * phi, randdir.x, randdir.y);
	int _octaves = SEARCH_OCTAVES + (level >= 1 ? 4 : 0);

	while(_octaves-- > 0)
	{
		_octaves = best_sim < 0.999999 ? _octaves : 0;
		float2 local_motion = 0;

		int _samples = OCTAVE_SAMPLES;
		while(_samples-- > 0)		
		{
			_samples = best_sim < 0.999999 ? _samples : 0;
			randdir = Math::rotate_2D(randdir, rotator);
			float2 search_offset = randdir * texelsize;
			float2 search_center = i.uv + total_motion + search_offset;			 

			m_yy = 0;
			m_xy = 0;

			[loop]
			for(uint k = 0; k < blocksize * blocksize; k++)
			{
				FEATURE_TYPE t = get_prev_feature(search_center + float2(k % blocksize, k / blocksize) * texelsize * search_scale, level).FEATURE_COMPS;
				m_yy += t * t;
				m_xy += local_block[k] * t;
			}
			float sim = minc(m_xy * rsqrt(m_xx * m_yy));
			if(sim < best_sim) continue;
			
			best_sim = sim;
			local_motion = search_offset;	
							
		}
		total_motion += local_motion;
		randdir *= 0.5;
	}

	
	float4 curr_layer = float4(total_motion, variance, saturate(-acos(best_sim) * PI + 1.0));
	return curr_layer;
}

float4 harmonics_matching(VSOUT i, int level, float4 coarse_layer, const int blocksize)
{	
	level = max(level - 1, 0); //sample one higher res for better quality
	float2 texelsize = rcp(tex2Dsize(sOldFeatureLayerPyramid, max(0, level - MIN_MIP)));

	float2 total_motion = coarse_layer.xy;
	float coarse_sim = coarse_layer.w;

	float4 local_harmonics[2];
	local_harmonics[0] = tex2Dlod(sOldCircularHarmonicsPyramidCurr0, saturate(i.uv), max(0, level - MIN_MIP));
	local_harmonics[1] = tex2Dlod(sOldCircularHarmonicsPyramidCurr1, saturate(i.uv), max(0, level - MIN_MIP));

	float4 search_harmonics[2];
	search_harmonics[0] = tex2Dlod(sOldCircularHarmonicsPyramidPrev0, saturate(i.uv + total_motion), max(0, level - MIN_MIP));
	search_harmonics[1] = tex2Dlod(sOldCircularHarmonicsPyramidPrev1, saturate(i.uv + total_motion), max(0, level - MIN_MIP));

	float m_xx = dot(1, local_harmonics[0]*local_harmonics[0] + local_harmonics[1]*local_harmonics[1]);
	float m_yy = dot(1, search_harmonics[0]*search_harmonics[0] + search_harmonics[1]*search_harmonics[1]);
	float m_xy = dot(1, search_harmonics[0]*local_harmonics[0] + search_harmonics[1]*local_harmonics[1]);

	float best_sim = m_xy * rsqrt(m_xx * m_yy);

	//this fixes completely white areas from polluting the buffer with false offsets
	if(best_sim > 0.999999) 
		return float4(coarse_layer.xy, 0, 0);

	float phi = radians(360.0 / OCTAVE_SAMPLES);
	float4 rotator = Math::get_rotator(phi);	
	float randseed = jitter(i.vpos.xy).x;
	randseed = QMC::roberts1(level, randseed);

	float2 randdir; sincos(randseed * phi, randdir.x, randdir.y);
	int _octaves = SEARCH_OCTAVES + (level >= 1 ? 2 : 0);

	while(_octaves-- > 0)
	{
		_octaves = best_sim < 0.999999 ? _octaves : 0;
		float2 local_motion = 0;

		int _samples = OCTAVE_SAMPLES;
		while(_samples-- > 0)		
		{
			_samples = best_sim < 0.999999 ? _samples : 0;
			randdir = Math::rotate_2D(randdir, rotator);
			float2 search_offset = randdir * texelsize;
			float2 search_center = i.uv + total_motion + search_offset;

			search_harmonics[0] = tex2Dlod(sOldCircularHarmonicsPyramidPrev0, saturate(search_center), max(0, level - MIN_MIP));
			search_harmonics[1] = tex2Dlod(sOldCircularHarmonicsPyramidPrev1, saturate(search_center), max(0, level - MIN_MIP));

			float m_yy = dot(1, search_harmonics[0]*search_harmonics[0] + search_harmonics[1]*search_harmonics[1]);
			float m_xy = dot(1, search_harmonics[0]*local_harmonics[0] + search_harmonics[1]*local_harmonics[1]);

			float sim = m_xy * rsqrt(m_xx * m_yy);
	
			if(sim < best_sim) continue;
			
			best_sim = sim;
			local_motion = search_offset;
							
		}
		total_motion += local_motion;
		randdir *= 0.5;
	}
	
	float4 curr_layer = float4(total_motion, 1.0, saturate(-acos(best_sim) * PI + 1.0));
	return curr_layer;
}

float4 atrous_upscale(VSOUT i, int level, sampler sMotionLow, int rad)
{	
    float2 texelsize = rcp(tex2Dsize(sMotionLow));
	float phi = QMC::roberts1(level * 0.2144, FRAMECOUNT % 16) * HALF_PI;
	float4 kernelmat = Math::get_rotator(phi) * FILTER_RADIUS;

	float4 sum = 0;
	float wsum = 1e-6;

	[loop]for(int x = -rad; x <= rad; x++)
	[loop]for(int y = -rad; y <= rad; y++)
	{
		float2 offs = float2(x, y);
		offs *= abs(offs);	
		float2 sample_uv = i.uv + Math::rotate_2D(offs, kernelmat) * texelsize; 

		float4 sample_gbuf = tex2Dlod(sMotionLow, sample_uv, 0);
		float2 sample_mv = sample_gbuf.xy;
		float sample_var = sample_gbuf.z;
		float sample_sim = sample_gbuf.w;

		float vv = dot(sample_mv, sample_mv);
		float ws = saturate(1.0 - sample_sim); ws *= ws;
		float wf = saturate(1 - sample_var * 128.0) * 3;
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;
		float weight = exp2(-(ws + wm + wf) * 4);

		weight *= all(saturate(sample_uv - sample_uv * sample_uv));
		sum += sample_gbuf * weight;
		wsum += weight;		
	}

	sum /= wsum;
	return sum;	
}

float4 atrous_upscale_temporal(VSOUT i, int level, sampler sMotionLow, int rad)
{	
   	float2 texelsize = rcp(tex2Dsize(sMotionLow));
	float phi = QMC::roberts1(level * 0.2144, FRAMECOUNT % 16) * HALF_PI;
	float4 kernelmat = Math::get_rotator(phi) * FILTER_RADIUS;

	float4 sum = 0;
	float wsum = 1e-6;

	float center_z = get_curr_feature(i.uv, max(0, level - 2)).y;

	[loop]for(int x = -rad; x <= rad; x++)
	[loop]for(int y = -rad; y <= rad; y++)
	{
		float2 offs = float2(x, y);
		offs *= abs(offs);	
		float2 sample_uv = i.uv + Math::rotate_2D(offs, kernelmat) * texelsize; 

		float4 sample_gbuf = tex2Dlod(sMotionLow, sample_uv, 0);
		float2 sample_mv = sample_gbuf.xy;
		float sample_var = sample_gbuf.z;
		float sample_sim = sample_gbuf.w;

		float vv = dot(sample_mv, sample_mv);

		float2 prev_mv = tex2Dlod(sOldMotionTexIntermediate0, sample_uv + sample_gbuf.xy, 0).xy;
		float2 mvdelta = prev_mv - sample_mv;
		float diff = dot(mvdelta, mvdelta) * rcp(1e-8 + max(vv, dot(prev_mv, prev_mv)));

		float wd = 3.0 * diff;		
		float ws = saturate(1.0 - sample_sim); ws *= ws;
		float wf = saturate(1 - sample_var * 128.0) * 3;
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;

		float weight = exp2(-(ws + wm + wf + wd) * 4);		

		float sample_z = get_curr_feature(sample_uv, max(0, level - 2)).y;
		float wz = abs(center_z - sample_z) / max3(0.00001, center_z, sample_z);
		weight *= exp2(-wz * 4);

		weight *= all(saturate(sample_uv - sample_uv * sample_uv));
		sum += sample_gbuf * weight;
		wsum += weight;
	}

	sum /= wsum;
	return sum;	
}

float4 motion_pass(in VSOUT i, sampler sMotionLow, int level, int filter_size, int block_size)
{
	float4 prior_motion = 0;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;
	
	return block_matching(i, level, prior_motion, block_size);	
}

float4 motion_pass_with_temporal_filter(in VSOUT i, sampler sMotionLow, int level, int filter_size, int block_size)
{
	float4 prior_motion = 0;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale_temporal(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;

	return block_matching(i, level, prior_motion, block_size);	
}

float4 motion_pass_with_temporal_filter_harmonics(in VSOUT i, sampler sMotionLow, int level, int filter_size, int block_size)
{
	float4 prior_motion = 0;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale_temporal(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;

	return harmonics_matching(i, level, prior_motion, block_size);
}

float4 motion_pass_harmonics(in VSOUT i, sampler sMotionLow, int level, int filter_size, int block_size)
{
	float4 prior_motion = 0;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;

	return harmonics_matching(i, level, prior_motion, block_size);
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(dist * 400));
}

float2 centroid_dir(float2 uv)
{
	const float2 offs[16] = {float2(-1, -3),float2(0, -3),float2(1, -3),float2(2, -2),float2(3, -1),float2(3, 0),float2(3, 1),float2(2,2),float2(1,3),float2(0,3),float2(-1,3),float2(-2,2),float2(-3,1),float2(-3,0),float2(-3,-1),float2(-2,-2)};
	float4 moments = 0;
	[unroll]
	for(int j = 0; j < 16; j++)
	{
		float v = dot(float3(0.299, 0.587, 0.114), tex2D(ColorInput, uv + BUFFER_PIXEL_SIZE * offs[j]).rgb);
		moments += float4(1, offs[j].x, offs[j].y, offs[j].x*offs[j].y) * v;
	}
	moments /= 16.0;
	return moments.yz / moments.x;
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

void WriteFeaturePS(in VSOUT i, out FEATURE_TYPE o : SV_Target0)
{	
	float4 feature_data = 0;
#if MIN_MIP > 0	
	const float4 radius = float4(0.7577, -0.7577, 2.907, 0);
	const float2 weight = float2(0.37487566, -0.12487566);
	feature_data.rgb =  weight.x * tex2D(ColorInput, i.uv + radius.xx * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.xy * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.yx * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.yy * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv + radius.zw * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv - radius.zw * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv + radius.wz * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv - radius.wz * BUFFER_PIXEL_SIZE).xyz;	
#else	
	feature_data.rgb = tex2D(ColorInput, i.uv).rgb;
#endif	
	feature_data.w = Depth::get_linear_depth(i.uv);	

#if OPTICAL_FLOW_MATCHING_LAYERS == 0
	o = dot(float3(0.299, 0.587, 0.114), feature_data.rgb);
#else
	o.x = dot(float3(0.299, 0.587, 0.114), feature_data.rgb);
	o.y = feature_data.w;
#endif
}

void generate_circular_harmonics(in VSOUT i, sampler s_features, out float4 coeffs0, out float4 coeffs1)
{
	float2 Y1 = 0;
	float2 Y2 = 0;
	float2 Y3 = 0;
	float2 Y4 = 0;

	//take 5 minutes writing down all the coefficients? no
	//take 15 minutes generating them procedurally? hell yes
	float2 gather_offsets[4] = {float2(-0.5, 0.5), float2(0.5, 0.5), float2(0.5, -0.5), float2(-0.5, -0.5)};
	float4 rotator = Math::get_rotator(radians(90.0));
	float2 offsets[3] = {float2(1.5, 0.5), float2(1.5, 2.5), float2(0.0, 3.5)};

	[unroll]
	for(int j = 0; j < 4; j++)
	{
		[unroll]
		for(int k = 0; k < 3; k++)
		{
			float2 p = offsets[k];
			float4 v = tex2DgatherR(s_features, i.uv + p * BUFFER_PIXEL_SIZE * exp2(MIN_MIP));

			[unroll]
			for(int ch = 0; ch < 4; ch++)
			{
				float x = p.x + gather_offsets[ch].x;
				float y = p.y + gather_offsets[ch].y;
				float r = sqrt(x*x+y*y+1e-6);

				Y1 += v[ch].xx * float2(y, x) / r;
				Y2 += v[ch].xx * float2(x * y, x*x - y*y) / (r * r);
				Y3 += v[ch].xx * float2(y*(3*x*x-y*y), x*(x*x-3*y*y)) / (r*r*r);
				Y4 += v[ch].xx * float2(x*y*(x*x-y*y),x*x*(x*x-3*y*y)-y*y*(3*x*x-y*y)) / (r*r*r*r);
			}

			offsets[k] = Math::rotate_2D(offsets[k], rotator);
		}
	}

	coeffs0 = float4(Y1, Y2);
	coeffs1 = float4(Y3, Y4);
}

void CircularHarmonicsPS(in VSOUT i, out PSOUT2 o)
{	
	generate_circular_harmonics(i, sOldFeatureLayerPyramid, o.t0, o.t1);
}

void CircularHarmonicsPrevPS(in VSOUT i, out PSOUT2 o)
{	
	generate_circular_harmonics(i, sOldFeatureLayerPyramidPrev, o.t0, o.t1);
}

void MotionPS6(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sOldMotionTexIntermediate2, 6, 2, 4);}
void MotionPS5(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sOldMotionTexIntermediate6, 5, 2, 4);}
void MotionPS4(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sOldMotionTexIntermediate5, 4, 2, 4);}
void MotionPS3(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sOldMotionTexIntermediate4, 3, 2, 4);}
#if OPTICAL_FLOW_MATCHING_LAYERS == 2
void MotionPS2(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter_harmonics(i, sOldMotionTexIntermediate3, 2, 2, 4);}
void MotionPS1(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_harmonics(i, sOldMotionTexIntermediate2, 1, 1, 3);}
void MotionPS0(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_harmonics(i, sOldMotionTexIntermediate1, 0, 1, 2);}
#else 
void MotionPS2(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sOldMotionTexIntermediate3, 2, 2, 4);}
void MotionPS1(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sOldMotionTexIntermediate2, 1, 1, 3);}
void MotionPS0(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sOldMotionTexIntermediate1, 0, 1, 2);}
#endif

#if VELOCITY_DEBUG_OUTPUT != 0
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
    ui_label = "iMMERSE Velocity Old (enable and move to the top!)";
    ui_tooltip =        
        "                           MartysMods - Velocity                              \n"
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
#if OPTICAL_FLOW_MATCHING_LAYERS == 2
	pass {VertexShader = MainVS;PixelShader  = CircularHarmonicsPrevPS;  RenderTarget0 = OldCircularHarmonicsPyramidPrev0; RenderTarget1 = OldCircularHarmonicsPyramidPrev1; }	
#endifgg
    pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; RenderTarget = OldFeatureLayerPyramid; } 
#if OPTICAL_FLOW_MATCHING_LAYERS == 2
	pass {VertexShader = MainVS;PixelShader  = CircularHarmonicsPS;  RenderTarget0 = OldCircularHarmonicsPyramidCurr0; RenderTarget1 = OldCircularHarmonicsPyramidCurr1; }
#endif
	pass {VertexShader = MainVS;PixelShader = MotionPS6;RenderTarget = OldMotionTexIntermediate6;}
    pass {VertexShader = MainVS;PixelShader = MotionPS5;RenderTarget = OldMotionTexIntermediate5;}
    pass {VertexShader = MainVS;PixelShader = MotionPS4;RenderTarget = OldMotionTexIntermediate4;}
    pass {VertexShader = MainVS;PixelShader = MotionPS3;RenderTarget = OldMotionTexIntermediate3;}
    pass {VertexShader = MainVS;PixelShader = MotionPS2;RenderTarget = OldMotionTexIntermediate2;}
    pass {VertexShader = MainVS;PixelShader = MotionPS1;RenderTarget = OldMotionTexIntermediate1;}
    pass {VertexShader = MainVS;PixelShader = MotionPS0;RenderTarget = OldMotionTexIntermediate0;}
	pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; RenderTarget = OldFeatureLayerPyramidPrev; }
#if VELOCITY_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }		
#endif 

}