use std::{collections::VecDeque, sync::Arc};

use nalgebra::RealField;
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

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum CurveVx {
    Vec3(na::Vec3),
    Curve { curve: CurveId, t: f32 },
}

pub type CurveFn = Arc<dyn Fn(f32) -> na::Vec3 + Send + Sync + 'static>;

lazy_static::lazy_static! {
    static ref ZERO_CURVE_FN: CurveFn = {
        let f = |t: f32| na::vec3(0.0, 0.0, t);
        Arc::new(f)
    };
}

pub struct CurveLayout {
    vertices: Vec<na::Vec3>,

    curve_endpoints: Vec<(usize, usize)>,
    curve_fns: Vec<CurveFn>,
    // curve_samples: Vec<LineSampler>,
}

impl CurveLayout {
    pub fn new_curve(&mut self, p0: na::Vec3, p1: na::Vec3) -> CurveId {
        let i = self.curve_endpoints.len();

        let vi = self.vertices.len();
        self.vertices.push(p0);
        self.vertices.push(p1);

        self.curve_endpoints.push((vi, vi + 1));
        self.curve_fns.push(ZERO_CURVE_FN.clone());

        CurveId(i)
    }

    pub fn add_child_curve(
        &mut self,
        parent: CurveId,
        t0: f32,
        t1: f32,
    ) -> CurveId {
        let (start_i, end_i) = self.curve_endpoints[parent.0];

        let p_p0 = self.vertices[start_i];
        let p_p1 = self.vertices[end_i];

        let cfn = &self.curve_fns[parent.0];

        let parent_mat = Self::transformation_matrix(p_p0, p_p1);

        let c_t0 = p_p0 + parent_mat * cfn(t0);
        let c_t1 = p_p0 + parent_mat * cfn(t1);

        let child_i = self.curve_endpoints.len();

        let vi = self.vertices.len();
        self.vertices.push(c_t0);
        self.vertices.push(c_t1);

        self.curve_endpoints.push((vi, vi + 1));
        self.curve_fns.push(ZERO_CURVE_FN.clone());

        CurveId(child_i)
    }

    // pub fn transformation_matrix(v0: na::Vec3, v1: na::Vec3) -> na::Mat4 {
    pub fn transformation_matrix(p0: na::Vec3, p1: na::Vec3) -> na::Mat3 {
        // #[rustfmt::skip]
        // let rot_xy = |d: f32| na::mat3(d.cos(), -d.sin(), 0.0,
        //                                d.sin(),  d.cos(), 0.0,
        //                                0.0,         0.0,  1.0);

        #[rustfmt::skip]
        let rot_xz = |d: f32| na::mat3(d.cos(), 0.0, -d.sin(),
                                       0.0,     1.0,      0.0,
                                       d.sin(), 0.0,  d.cos());

        #[rustfmt::skip]
        let rot_yz = |d: f32| na::mat3(1.0,     0.0,      0.0,
                                       0.0, d.cos(), -d.sin(),
                                       0.0, d.sin(),  d.cos());

        let cdel = p1 - p0;

        let theta = cdel.z.atan2(cdel.x);
        let alpha = cdel.y.atan2(cdel.z);

        let z_scale = p1.metric_distance(&p1);

        let rots = rot_yz(alpha) * rot_xz(theta);

        #[rustfmt::skip]
        let scale = na::mat3(1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, z_scale);

        let mat = rots * scale;

        mat
    }

    /*
    pub fn reify(&self) -> (Vec<na::Vec3>, Vec<Vec<usize>>) {
        let curve_n = self.curve_endpoints.len();

        let mut vertices = Vec::new();
        let mut lines = Vec::new();

        for curve_ix in 0..curve_n {
            let (start_i, end_i) = self.curve_endpoints[curve_ix];

            let p0 = self.vertices[start_i];
            let p1 = self.vertices[end_i];

            let mat = Self::transformation_matrix(p0, p1);

            // let translation = na::translation(p0);
        }
    }
    */
}

#[derive(Default)]
pub struct CurveNetwork {
    // pub vertices: Vec<na::Vec3>,
    vertices: Vec<CurveVx>,

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

        let min_vx_count = 15;

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

        #[rustfmt::skip]
        let rot_xy = |d: f32| na::mat3(d.cos(), -d.sin(), 0.0,
                                       d.sin(),  d.cos(), 0.0,
                                       0.0,         0.0,  1.0);

        #[rustfmt::skip]
        let rot_xz = |d: f32| na::mat3(d.cos(), 0.0, -d.sin(),
                                       0.0,     1.0,      0.0,
                                       d.sin(), 0.0,  d.cos());

        #[rustfmt::skip]
        let rot_yz = |d: f32| na::mat3(1.0,     0.0,      0.0,
                                       0.0, d.cos(), -d.sin(),
                                       0.0, d.sin(),  d.cos());

        let del = end - start;

        let rot = rot_yz(del.y.atan2(del.z)) * rot_xz(del.z.atan2(del.x));

        let len = del.norm();

        #[rustfmt::skip]
        let mat = na::mat3(1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, len);

        let mat = mat * rot;
        // let mat = mat * rot;
        // let mat = mat * rot.transpose();

        let pre = (self.curve_fns[curve.0])(t);

        // let pre =

        let out = start + (mat * pre);

        Some(out)
    }

    pub fn curve_endpoints(&self, curve: CurveId) -> (VxVar, VxVar) {
        self.curve_endpoints[curve.0]
    }

    pub fn new_vertex(&mut self) -> VxId {
        let i = self.vertices.len();
        self.vertices.push(CurveVx::Vec3(na::zero()));
        VxId(i)
    }

    pub fn set_vertex(&mut self, i: VxId, new: na::Vec3) {
        self.vertices[i.0] = CurveVx::Vec3(new);
    }

    // pub fn set_vertex_to_curve(&mut self, i: VxId, curve: CurveId, t: f32) {
    //     self.vertices[i.0] = CurveVx::Curve { curve, t };
    // }

    pub fn set_vertex_to_curve(&mut self, i: VxId, curve: CurveId, t: f32) {
        let v = (&self.curve_fns[curve.0])(t);
        self.vertices[i.0] = CurveVx::Vec3(v);
        // self.vertices[i.0] = CurveVx::Curve { curve, t };
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

    // only returns "concrete" vertices, not ones defined relative to
    // a curve
    pub fn get_vx(&self, v: VxId) -> Option<na::Vec3> {
        match self.vertices[v.0] {
            CurveVx::Vec3(v) => Some(v),
            CurveVx::Curve { curve, t } => {
                let (start, end) = self.curve_endpoints(curve);
                if t == 0.0 {
                    self.read_vx_var(start)
                } else if t == 1.0 {
                    self.read_vx_var(end)
                } else {
                    // let s = self.read_vx_var(start)?;
                    // let e = self.read_vx_var(end)?;
                    let curve = &self.curve_fns[curve.0];
                    Some(curve(t))
                }
            }
        }
    }

    pub fn read_vx_var(&self, var: VxVar) -> Option<na::Vec3> {
        let ix = *self.vertex_variables.get(var.0)?;
        let ix = ix?;
        let v = self.vertices.get(ix).copied()?;
        match v {
            CurveVx::Vec3(v) => Some(v),
            CurveVx::Curve { .. } => None,
        }
    }

    pub fn is_assigned(&self, var: VxVar) -> bool {
        self.vertex_variables
            .get(var.0)
            .map(|v| v.is_some())
            .unwrap_or_default()
    }

    pub fn reify(&self) -> (Vec<na::Vec3>, Vec<Vec<usize>>) {
        let curve_n = self.curve_endpoints.len();

        let mut curve_dequeue: VecDeque<usize> = VecDeque::new();

        for curve_ix in 0..curve_n {
            curve_dequeue.push_back(curve_ix);
        }

        let mut vertices = Vec::new();
        let mut lines = Vec::new();

        while let Some(curve_ix) = curve_dequeue.pop_front() {
            let (start, end) = self.curve_endpoints[curve_ix];

            // only reify curves with actual vertex endpoints
            if !self.is_assigned(start) || !self.is_assigned(end) {
                continue;
            }
            /*
            else if !self.is_concrete(start) || !self.is_concrete(end) {
                // if one of the endpoints haven't been assigned, do
                // this curve later
                eprintln!("pushing curve {:?}", curve_ix);
                curve_dequeue.push_back(curve_ix);
            }
            */

            let v0 = self.read_vx_var(start).unwrap_or_default();
            let v1 = self.read_vx_var(end).unwrap_or_default();

            let mut cur_line = Vec::new();
            // let sampler = &self.curve_samples[curve_ix];

            let i = vertices.len() + 1;

            cur_line.push(i);
            cur_line.push(i + 1);

            vertices.push(v0);
            vertices.push(v1);

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
