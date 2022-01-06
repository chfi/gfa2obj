use std::sync::Arc;

use nalgebra_glm as na;

#[derive(Clone)]
pub struct LineSampler {
    min_sample_dist: f32,
    fixed_samples: Vec<f32>,
    offsets: Vec<f32>,
}

impl LineSampler {
    pub fn sample_points(&self) -> &[f32] {
        &self.offsets
    }

    pub fn sample(&self, t: f32) -> f32 {
        if t == 0.0 || t == 1.0 {
            return t;
        }

        let i = self.sample_rank(t);
        self.offsets[i]
    }

    pub fn sample_rank(&self, t: f32) -> usize {
        let i_res = self
            .offsets
            .binary_search_by(|v| v.partial_cmp(&t).unwrap());

        match i_res {
            Ok(i) => i,
            Err(i) => i.clamp(0, self.offsets.len() - 1),
        }
    }

    fn build_offsets(min_dist: f32, fixed: &[f32], out: &mut Vec<f32>) {
        out.clear();

        let mut cur_t = 0.0;

        out.push(cur_t);

        for &pt in fixed {
            assert!(cur_t <= 1.0);

            let remaining = (pt - cur_t).abs();

            let regulars = (remaining / min_dist).floor() as usize;

            for _i in 0..regulars {
                out.push(cur_t);
                cur_t += min_dist;
            }

            out.push(pt);
        }

        out.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        out.dedup();
    }

    pub fn from_min_vertex_count(n: usize) -> Self {
        let n = n.max(2) as f32;

        let min_sample_dist = 1.0 / n;

        // let fixed_samples = vec![];
        let fixed_samples = vec![0.0, 0.5, 1.0];

        let mut offsets = Vec::new();

        Self::build_offsets(min_sample_dist, &fixed_samples, &mut offsets);

        Self {
            min_sample_dist,
            fixed_samples,
            offsets,
        }
    }

    pub fn push_samples(&mut self, samples: impl Iterator<Item = f32>) {
        self.fixed_samples.extend(samples);
        self.fixed_samples
            .sort_by(|a, b| a.partial_cmp(&b).unwrap());
        self.fixed_samples.dedup();

        Self::build_offsets(
            self.min_sample_dist,
            &self.fixed_samples,
            &mut self.offsets,
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VxId(usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VxVar(usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CurveId(usize);

pub type CurveFn = Arc<dyn Fn(f32) -> na::Vec3 + Send + Sync + 'static>;

lazy_static::lazy_static! {
    static ref ZERO_CURVE_FN: CurveFn = {
        let f = |t: f32| na::vec3(0.0, 0.0, t);
        Arc::new(f)
    };
}

pub struct CurveNetwork {
    pub vertices: Vec<na::Vec3>,

    vertex_variables: Vec<Option<usize>>,

    curve_endpoints: Vec<(VxVar, VxVar)>,
    curve_fns: Vec<CurveFn>,
    curve_samples: Vec<LineSampler>,
    // curve_transforms: Vec<na::Mat3>,
}

impl CurveNetwork {
    pub fn new_curve(&mut self) -> CurveId {
        let i = self.curve_endpoints.len();

        let p0_ = self.new_vertex_var();
        let p1_ = self.new_vertex_var();

        self.curve_endpoints.push((p0_, p1_));
        self.curve_fns.push(ZERO_CURVE_FN.clone());

        let min_vx_count = 5;

        self.curve_samples
            .push(LineSampler::from_min_vertex_count(min_vx_count));

        CurveId(i)
    }

    pub fn set_curve_fn(&mut self, curve: CurveId, f: CurveFn) {
        self.curve_fns[curve.0] = f;
    }

    // returns None if one of the endpoints haven't been assigned yet
    pub fn sample_curve(&self, curve: CurveId, t: f32) -> Option<na::Vec3> {
        let (s, e) = self.curve_endpoints[curve.0];

        let start = self.read_vx_var(s)?;
        let end = self.read_vx_var(e)?;

        let len = (end - start).norm();

        #[rustfmt::skip]
        let mat = na::mat3(1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, len);

        let pre = (self.curve_fns[curve.0])(t);

        Some(mat * pre)
    }

    pub fn curve_endpoints(&self, curve: CurveId) -> (VxVar, VxVar) {
        self.curve_endpoints[curve.0]
    }

    pub fn new_vertex(&mut self) -> VxId {
        let i = self.vertices.len();
        self.vertices.push(na::zero());
        VxId(i)
    }

    pub fn set_vertex(&mut self, i: VxId, new: na::Vec3) {
        self.vertices[i.0] = new;
    }

    pub fn new_vertex_var(&mut self) -> VxVar {
        let i = self.vertex_variables.len();
        self.vertex_variables.push(None);
        VxVar(i)
    }

    pub fn assign(&mut self, var: VxVar, vx: VxId) -> Option<VxId> {
        if let Some(var) = self.vertex_variables.get_mut(var.0) {
            let old = var.map(|v| VxId(v));
            *var = Some(vx.0);
            old
        } else {
            panic!(
                "attempted to assign to nonexistent vertex variable {}",
                var.0
            );
        }
    }

    pub fn read_vx_var(&self, var: VxVar) -> Option<na::Vec3> {
        let ix = *self.vertex_variables.get(var.0)?;
        let ix = ix?;
        self.vertices.get(ix).copied()
    }

    pub fn is_assigned(&self, var: VxVar) -> bool {
        self.vertex_variables
            .get(var.0)
            .map(|v| v.is_some())
            .unwrap_or_default()
    }

    pub fn reify(&self) -> (Vec<na::Vec3>, Vec<Vec<usize>>) {
        let curve_n = self.curve_endpoints.len();

        let mut vertices = Vec::new();
        let mut lines = Vec::new();

        for curve_ix in 0..curve_n {
            let (start, end) = self.curve_endpoints[curve_ix];

            // only reify curves with actual vertex endpoints
            if !self.is_assigned(start) || !self.is_assigned(end) {
                continue;
            }

            let mut cur_line = Vec::new();

            let curve_fn = &self.curve_fns[curve_ix];
            let sampler = &self.curve_samples[curve_ix];
            // let mat = &self.curve_transforms[curve_ix];

            let ts = sampler.sample_points();

            for &t in ts {
                let p = curve_fn(t);
                let i = vertices.len();
                vertices.push(p);
                cur_line.push(i + 1);
            }

            lines.push(cur_line);
        }

        (vertices, lines)
    }

    pub fn write_obj<W: std::io::Write>(
        &self,
        mut out: W,
    ) -> anyhow::Result<()> {
        let (vertices, lines) = self.reify();

        for v in vertices {
            writeln!(out, "v {} {} {}", v.x, v.y, v.z)?;
        }

        for line in lines {
            write!(out, "l")?;
            for ix in line {
                write!(out, " {}", ix)?;
            }
            writeln!(out)?;
        }

        Ok(())
    }
}
