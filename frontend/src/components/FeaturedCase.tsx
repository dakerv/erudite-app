import React from 'react';
import { FEATURED_IMAGE } from '../data/content';

export function FeaturedCase() {
  return (
    <section
      aria-labelledby="featured-headline"
      className="border-b border-rule">
      
      <div className="mx-auto max-w-spread px-5 py-12 sm:px-8 lg:px-12 lg:py-16">
        <div className="flex items-baseline justify-between border-b border-ink pb-2">
          <p className="font-mono text-[10px] uppercase tracking-label text-ink">
            Case study / Face-swapped
          </p>
          <p className="font-mono text-[10px] uppercase tracking-label text-gray-soft">
            Plate 01
          </p>
        </div>

        <figure className="mt-5">
          <div className="paper-grain relative overflow-hidden bg-ivorydeep">
            <img
              src={FEATURED_IMAGE}
              alt="Press conference photograph in which the speaker's face has been digitally replaced."
              className="h-[280px] w-full object-cover object-[center_28%] sm:h-[400px] lg:h-[520px]" />
            
          </div>
          <figcaption className="mt-3 grid gap-4 border-t border-rule pt-3 sm:grid-cols-12">
            <p className="font-mono text-[10px] uppercase leading-[1.6] tracking-label text-gray-soft sm:col-span-5">
              Fig. 01 — Municipal press briefing, undated. Facial region
              replaced; lighting and grain preserved.
            </p>
            <p className="font-mono text-[10px] uppercase tracking-label text-gray-soft sm:col-span-7 sm:text-right">
              Submitted for examination · Classified face-swapped at 96.1%
            </p>
          </figcaption>
        </figure>

        <div className="mt-10 grid gap-8 lg:grid-cols-12 lg:gap-14">
          <h2
            id="featured-headline"
            className="font-display uppercase leading-[0.95] tracking-[-0.015em] text-ink lg:col-span-7"
            style={{ fontSize: 'clamp(2.1rem, 4.6vw, 4.2rem)' }}>
            
            Do you know this
            <br />
            is <span className="italic lowercase">synthetic?</span>
          </h2>
          <div className="lg:col-span-5 lg:pt-2">
            <p className="max-w-[48ch] text-[16px] leading-[1.65] text-graphite">
              Modern generation techniques can produce images that appear
              ordinary at first glance, making visual authenticity increasingly
              difficult to judge.
            </p>
          </div>
        </div>
      </div>
    </section>);

}