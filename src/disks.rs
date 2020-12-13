use libm::{erf, exp, sqrt};
use std::f64::consts::PI;




static G: f64 = 1.0; // gravitational constant
static M: f64 = 1.0; // system mass




// ============================================================================
pub struct Torus {
	pub mach_number: f64,
	pub softening_length: f64,
	pub mass: f64,
	pub radius: f64,
	pub width: f64,
}




// ============================================================================
impl Torus {

	pub fn real_mach_number_squared(&self, r: f64) -> f64 {
		let ma = self.mach_number;
		let rs = self.softening_length;
		ma * ma - ((r * r - 2.0 * rs * rs) / (r * r + rs * rs) - self.dlogrho_dlogr(r))
	}

	pub fn phi_velocity_squared(&self, r: f64) -> f64 {
		self.sound_speed_squared(r) * self.real_mach_number_squared(r)
	}

	pub fn vertically_integrated_pressure(&self, r: f64) -> f64 {
		self.surface_density(r) * self.sound_speed_squared(r)
	}

	pub fn kepler_speed_squared(&self, r: f64) -> f64 {
		let rs = self.softening_length;
		G * M * r * r / (r * r + rs * rs).powf(1.5)
	}

	pub fn sound_speed_squared(&self, r: f64) -> f64 {
		self.kepler_speed_squared(r) / self.mach_number.powi(2)
	}

	pub fn surface_density(&self, r: f64) -> f64 {
		let r0 = self.radius;
		let dr = self.width;
		self.sigma0() * exp(-((r - r0) / dr).powi(2))
	}

	fn dlogrho_dlogr(&self, r: f64) -> f64 {
		let r0 = self.radius;
		let dr = self.width;
		-2.0 * r * (r - r0) / dr.powi(2)
	}

	fn sigma0(&self) -> f64 {
		let r0 = self.radius;
		let dr = self.width;
		let md = self.mass;
		let total = PI * dr * dr * (exp(-(r0 / dr).powi(2)) + sqrt(PI) * r0 / dr * (1.0 + erf(r0 / dr)));
		return md / total;
	}
}
